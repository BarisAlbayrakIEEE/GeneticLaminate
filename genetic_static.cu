

#include <stdio.h>
#include <cstddef>
#include <chrono>
#include <curand_kernel.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define PLY_COUNT 32
#define BLOCK_DIM0 32
#define BLOCK_DIM1 31
#define UINT64_ANGLE_0 0ui64
#define UINT64_ANGLE_P45 0xAAAAAAAAAAAAAAAAui64
#define UINT64_ANGLE_90 0xFFFFFFFFFFFFFFFFui64

#define ANGLE_INDEX_ARR_SIZE 4    // [0, 1, 2, 3]: [0, +45, -45, 90]
#define ANGLE_INDEX_0 0           // 0
#define ANGLE_INDEX_P45 1         // +45
#define ANGLE_INDEX_N45 2         // -45
#define ANGLE_INDEX_90 3          // 90
#define Q_ARR_SIZE 4              // Q11, Q12, Q22, Q66
#define QBAR_ARR_SIZE 5           // Qbar11, Qbar12, Qbar16, Qbar22, Qbar66
#define A_ARR_SIZE QBAR_ARR_SIZE  // A11, A12, A16, A22, A66
#define D_ARR_SIZE 3              // D11, D22, D66
#define ABD_ARR_SIZE 8            // A11, A12, A16, A22, A66, D11, D22, D66

typedef uint64_t _Stacking_t;
typedef unsigned char _Stacking_angle_index_t;

// Angle sins and cosines for: 0, 45, -45, 90 degrees
static const float C1S3[] = { 0.00f, 0.25f, -0.25f, 0.00f }; // cos^1 * sin^3
static const float C2S2[] = { 0.00f, 0.25f, 0.25f, 0.00f };  // cos^2 * sin^2
static const float C3S1[] = { 0.00f, 0.25f, -0.25f, 0.00f }; // cos^3 * sin^1
static const float C4[] = { 1.00f, 0.25f, 0.25f, 0.00f };    // cos^4
static const float S4[] = { 0.00f, 0.25f, 0.25f, 1.00f };    // sin^4

static const size_t STACK_SIZE_MAX_ELEMENT_1 = 64;
static const size_t STACK_SIZE_MAX_ELEMENT_2 = 80;
static const size_t STACK_SIZE_QUICK_SORT_1 = 80;
static const size_t STACK_SIZE_QUICK_SORT_2 = 96;
static const size_t STACK_SIZE_FUNCTION_COEFF = 2;
static const size_t STACK_SIZE_GLOBAL_SF = 4;
static const size_t CROSSOVER_TYPE_COUNT = 4;

__constant__ float As_cache__d[A_ARR_SIZE][ANGLE_INDEX_ARR_SIZE];
__constant__ float Ds_cache__d[D_ARR_SIZE][PLY_COUNT][ANGLE_INDEX_ARR_SIZE];
__constant__ float ABDs_treshold__d[ABD_ARR_SIZE];
__constant__ float fitness_coeffs__d[ABD_ARR_SIZE];

struct Orthotropic {
    float _t{ 0.184f };
    float _e11{ 162000.f };
    float _e22{ 8500.f };
    float _g12{ 4900.f };
    float _v12{ 0.34f };

    Orthotropic() = default;
    Orthotropic(
        float t,
        float e11,
        float e22,
        float g12,
        float v12)
        : _t(t), _e11(e11), _e22(e22), _g12(g12), _v12(v12) {};
};

void generate_cubic_difs(
    Orthotropic const* orthotropic,
    size_t size_,
    float* cubic_difs)
{
    size_t current{ size_ };
    size_t* cubes;
    cubes = (size_t*)malloc((size_ + 1) * sizeof(size_t));
    for (size_t i = 0; i < PLY_COUNT + 1; ++i) {
        cubes[i] = (size_t)pow(current--, 3);
    }

    float _2tcube3 = (float)(2 * pow(orthotropic->_t, 3) / 3.f);
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        cubic_difs[i] = _2tcube3 * (cubes[i] - cubes[i + 1]);
    }
};

void calculate_Qs(
    Orthotropic const* orthotropic,
    float* Qs)
{
    float temp{
        orthotropic->_e11 - orthotropic->_e22 * orthotropic->_v12 * orthotropic->_v12
    };
    Qs[0] = orthotropic->_e11 * orthotropic->_e11 / temp;                     // Q11
    Qs[1] = orthotropic->_v12 * orthotropic->_e11 * orthotropic->_e22 / temp; // Q12
    Qs[2] = orthotropic->_e11 * orthotropic->_e22 / temp;                     // Q22
    Qs[3] = orthotropic->_g12;                                                // Q66
};

void calculate_Qbars(
    float const Qs[Q_ARR_SIZE],
    unsigned char angle_index,
    float* Qbars)
{
    float Q11{ Qs[0] };
    float Q12{ Qs[1] };
    float Q22{ Qs[2] };
    float Q66{ Qs[3] };

    float c4_{ C4[angle_index] };
    float c1s3_{ C1S3[angle_index] };
    float c2s2_{ C2S2[angle_index] };
    float c3s1_{ C3S1[angle_index] };
    float s4_{ S4[angle_index] };

    float temp1 = (float)(2.f * (Q12 + 2.f * Q66) * c2s2_);
    float temp2 = (float)(Q11 - Q12 - 2.f * Q66);
    float temp3 = (float)(temp2 - Q11 + Q22);
    float temp4 = (float)(temp2 + temp3 + 2.f * Q12);
    float temp5 = (float)(temp2 + Q22 - Q12);

    Qbars[0] = Q11 * c4_ + temp1 + Q22 * s4_;     // Qbar11
    Qbars[1] = Q12 * (c4_ + s4_) + temp4 * c2s2_; // Qbar12
    Qbars[2] = temp2 * c3s1_ - temp3 * c1s3_;     // Qbar16
    Qbars[3] = Q11 * s4_ + temp1 + Q22 * c4_;     // Qbar22
    Qbars[4] = temp5 * c2s2_ + Q66 * (c4_ + s4_); // Qbar66
};

void calculate_Qbar_arr(
    float const Qs[Q_ARR_SIZE],
    float(*Qbar_arr)[QBAR_ARR_SIZE])
{
    calculate_Qbars(Qs, ANGLE_INDEX_0, Qbar_arr[0]);
    calculate_Qbars(Qs, ANGLE_INDEX_P45, Qbar_arr[1]);
    calculate_Qbars(Qs, ANGLE_INDEX_N45, Qbar_arr[2]);
    calculate_Qbars(Qs, ANGLE_INDEX_90, Qbar_arr[3]);
};

void calculate_As_cache(
    Orthotropic const* orthotropic,
    float const Qbar_arr[ANGLE_INDEX_ARR_SIZE][QBAR_ARR_SIZE],
    float(*As_cache__h)[ANGLE_INDEX_ARR_SIZE])
{
    float _2t{ 2 * orthotropic->_t };
    for (size_t i = 0; i < ANGLE_INDEX_ARR_SIZE; ++i) {
        float const* Qbars{ Qbar_arr[i] };
        for (size_t j = 0; j < A_ARR_SIZE; ++j) {
            As_cache__h[j][i] = Qbars[j] * _2t;
        }
    }
};

void calculate_Ds_cache(
    float const Qbar_arr[ANGLE_INDEX_ARR_SIZE][QBAR_ARR_SIZE],
    float const cubic_difs[PLY_COUNT],
    float(*Ds_cache__h)[PLY_COUNT][ANGLE_INDEX_ARR_SIZE])
{
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        float cubic_dif = cubic_difs[i];
        for (size_t j = 0; j < ANGLE_INDEX_ARR_SIZE; ++j) {
            float const* Qbars = Qbar_arr[j];
            Ds_cache__h[0][i][j] = Qbars[0] * cubic_dif; // D11
            Ds_cache__h[1][i][j] = Qbars[3] * cubic_dif; // D22
            Ds_cache__h[2][i][j] = Qbars[4] * cubic_dif; // D66
        }
    }
};

inline void get_stacking_angle_indexs__h(
    _Stacking_t stacking,
    _Stacking_angle_index_t* stacking_angle_indexs)
{
    size_t loc = PLY_COUNT;
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        --loc;
        stacking_angle_indexs[i] = (stacking >> (2 * loc)) & 0x3;
    }
};

void calculate_ABDs_treshold_helper(
    float const As_cache__h[A_ARR_SIZE][ANGLE_INDEX_ARR_SIZE],
    float const Ds_cache__h[D_ARR_SIZE][PLY_COUNT][ANGLE_INDEX_ARR_SIZE],
    _Stacking_t stacking,
    float* ABDs)
{
    _Stacking_angle_index_t stacking_angle_indexs[PLY_COUNT];
    get_stacking_angle_indexs__h(stacking, stacking_angle_indexs);
    size_t angle_index_counts[] = { 0, 0, 0, 0 };
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        ++angle_index_counts[stacking_angle_indexs[i]];
    }
    for (size_t i = 0; i < A_ARR_SIZE; ++i) {
        ABDs[i] = 0;
        for (size_t j = 0; j < ANGLE_INDEX_ARR_SIZE; ++j) {
            ABDs[i] += As_cache__h[i][j] * (float)angle_index_counts[j];
        }
    }
    for (size_t i = A_ARR_SIZE; i < ABD_ARR_SIZE; ++i) {
        ABDs[i] = 0;
        for (size_t j = 0; j < PLY_COUNT; ++j) {
            ABDs[i] += Ds_cache__h[i - A_ARR_SIZE][j][stacking_angle_indexs[j]];
        }
    }
};

void calculate_ABDs_treshold_main(
    float const As_cache__h[A_ARR_SIZE][ANGLE_INDEX_ARR_SIZE],
    float const Ds_cache__h[D_ARR_SIZE][PLY_COUNT][ANGLE_INDEX_ARR_SIZE],
    float* ABDs_treshold__h)
{
    float ABDs_uni[3][ABD_ARR_SIZE];
    calculate_ABDs_treshold_helper(As_cache__h, Ds_cache__h, UINT64_ANGLE_0, ABDs_uni[0]);
    calculate_ABDs_treshold_helper(As_cache__h, Ds_cache__h, UINT64_ANGLE_P45, ABDs_uni[1]);
    calculate_ABDs_treshold_helper(As_cache__h, Ds_cache__h, UINT64_ANGLE_90, ABDs_uni[2]);

    float treshold_ratio{ 4. };
    ABDs_treshold__h[0] = ABDs_uni[0][0] / treshold_ratio; // A11: stack of 0
    ABDs_treshold__h[1] = ABDs_uni[1][1] / treshold_ratio; // A12: stack of 45
    ABDs_treshold__h[2] = ABDs_uni[1][2] / treshold_ratio; // A16: stack of 45
    ABDs_treshold__h[3] = ABDs_uni[2][3] / treshold_ratio; // A22: stack of 90
    ABDs_treshold__h[4] = ABDs_uni[1][4] / treshold_ratio; // A66: stack of 45
    ABDs_treshold__h[5] = ABDs_uni[0][5] / treshold_ratio; // D11: stack of 0
    ABDs_treshold__h[6] = ABDs_uni[2][6] / treshold_ratio; // D22: stack of 90
    ABDs_treshold__h[7] = ABDs_uni[1][7] / treshold_ratio; // D66: stack of 45
};

void get_recursion_info_uniform(
    size_t stack_size_global,
    size_t population_size_function,
    size_t stack_size_function_1,
    size_t stack_size_function_2,
    size_t* recursion_limit,
    size_t* stack_size_thread_req)
{
    size_t n_thread = 2 * population_size_function;
    size_t stack_size_thread_available = (size_t)(stack_size_global / STACK_SIZE_GLOBAL_SF / n_thread);
    size_t recursion_depth = (size_t)log2(population_size_function);
    *stack_size_thread_req = STACK_SIZE_FUNCTION_COEFF * stack_size_function_1 * recursion_depth;
    *recursion_limit = 0;
    if (stack_size_thread_available >= *stack_size_thread_req) return;

    while (stack_size_thread_available < *stack_size_thread_req && *recursion_limit < recursion_depth) {
        ++*recursion_limit;
        n_thread = (size_t)(population_size_function / *recursion_limit);
        stack_size_thread_available = (size_t)(stack_size_global / STACK_SIZE_GLOBAL_SF / n_thread);
        recursion_depth = (size_t)log2(n_thread);
        *stack_size_thread_req = STACK_SIZE_FUNCTION_COEFF * stack_size_function_2 * recursion_depth;
    }
};

void get_recursion_info_nonuniform(
    size_t stack_size_global,
    size_t population_size_function,
    size_t stack_size_function_1,
    size_t stack_size_function_2,
    size_t* recursion_limit,
    size_t* stack_size_thread_req)
{
    size_t n_thread = 2 * population_size_function;
    size_t stack_size_thread_available = (size_t)(stack_size_global / STACK_SIZE_GLOBAL_SF / n_thread);
    size_t recursion_depth = population_size_function; // worst case
    *stack_size_thread_req = STACK_SIZE_FUNCTION_COEFF * stack_size_function_1 * recursion_depth;
    *recursion_limit = 0;
    if (stack_size_thread_available >= *stack_size_thread_req) return;

    while (stack_size_thread_available < *stack_size_thread_req && *recursion_limit < recursion_depth) {
        ++*recursion_limit;
        n_thread = (size_t)(2 * population_size_function - 2 * (*recursion_limit));
        stack_size_thread_available = (size_t)(stack_size_global / STACK_SIZE_GLOBAL_SF / n_thread);
        recursion_depth = population_size_function - *recursion_limit; // worst case
        *stack_size_thread_req = STACK_SIZE_FUNCTION_COEFF * stack_size_function_2 * recursion_depth;
    }
};

__global__ void initialize_indexs(
    size_t population_size_crossover,
    size_t* fitness_rate_indexs__d)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size_crossover) return;
    fitness_rate_indexs__d[tid] = tid;
};

[[deprecated]] __global__ void max_element_recursive_no_limit__d(
    float const* arr_vals__d,
    size_t low,
    size_t up,
    size_t* max_index__d,
    float* max_val__d)
{
    // base case
    if (low == up) {
        *max_index__d = low;
        *max_val__d = arr_vals__d[low];
        return;
    }

    // recursion
    size_t half = (low + up) / 2;
    size_t max_index1 = 0;
    size_t max_index2 = 0;
    float max_val1 = 0.f;
    float max_val2 = 0.f;
    max_element_recursive_no_limit__d<<<1, 1>>>(
        arr_vals__d,
        low,
        half,
        &max_index1,
        &max_val1);
    max_element_recursive_no_limit__d<<<1, 1>>>(
        arr_vals__d,
        half + 1,
        up,
        &max_index2,
        &max_val2);
    if (max_val1 > max_val2) {
        *max_index__d = max_index1;
        *max_val__d = max_val1;
    }
    else {
        *max_index__d = max_index2;
        *max_val__d = max_val2;
    }
};

[[deprecated]] __global__ void max_element_recursive_limited__d(
    float const* arr_vals__d,
    size_t low,
    size_t up,
    size_t recursion_limit,
    size_t* max_index__d,
    float* max_val__d)
{
    // base case
    if (up - low <= recursion_limit) {
        *max_index__d = low;
        *max_val__d = arr_vals__d[low];
        for (size_t i = low + 1; i <= up; ++i) {
            if (arr_vals__d[i] > *max_val__d) {
                *max_index__d = i;
                *max_val__d = arr_vals__d[i];
            }
        }
        return;
    }

    // recursion
    size_t half = (low + up) / 2;
    size_t max_index1 = 0;
    size_t max_index2 = 0;
    float max_val1 = 0.f;
    float max_val2 = 0.f;
    max_element_recursive_limited__d<<<1, 1>>>(
        arr_vals__d,
        low,
        half,
        recursion_limit,
        &max_index1,
        &max_val1);
    max_element_recursive_limited__d<<<1, 1>>>(
        arr_vals__d,
        half + 1,
        up,
        recursion_limit,
        &max_index2,
        &max_val2);
    if (max_val1 > max_val2) {
        *max_index__d = max_index1;
        *max_val__d = max_val1;
    }
    else {
        *max_index__d = max_index2;
        *max_val__d = max_val2;
    }
};

__global__ void max_element_no_recursion_helper__d(
    size_t const* arr_indexs__d,
    float const* arr_vals__d,
    size_t size_,
    size_t* max_indexs_temp,
    float* max_vals_temp)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_) return;

    size_t index1 = 2 * tid;
    size_t index2 = index1 + 1;
    if (arr_vals__d[index1] > arr_vals__d[index2]) {
        max_indexs_temp[tid] = arr_indexs__d[index1];
        max_vals_temp[tid] = arr_vals__d[index1];
    }
    else {
        max_indexs_temp[tid] = arr_indexs__d[index2];
        max_vals_temp[tid] = arr_vals__d[index2];
    }
};

__global__ void max_element_no_recursion_main__d(
    size_t* arr_indexs__d,
    float* arr_vals__d,
    size_t size_,
    size_t* arr_indexs_temp1__d,
    size_t* arr_indexs_temp2__d,
    float* arr_vals_temp1__d,
    float* arr_vals_temp2__d,
    size_t* max_index__d,
    float* max_val__d)
{
    size_t recursion_depth = (size_t)log2f(size_);
    size_t* arr_indexs_current__d = arr_indexs__d;
    float* arr_vals_current__d = arr_vals__d;
    size_t size_current = size_;
    for (size_t i = 0; i < recursion_depth / 2; ++i) {
        dim3 grid_dim1((size_current / 2 + BLOCK_DIM1) / BLOCK_DIM0);
        max_element_no_recursion_helper__d<<<grid_dim1, BLOCK_DIM0>>>(
            arr_indexs_current__d,
            arr_vals_current__d,
            size_current,
            arr_indexs_temp1__d,
            arr_vals_temp1__d);

        size_current /= 2;
        dim3 grid_dim2((size_current / 2 + BLOCK_DIM1) / BLOCK_DIM0);
        max_element_no_recursion_helper__d<<<grid_dim2, BLOCK_DIM0>>>(
            arr_indexs_temp1__d,
            arr_vals_temp1__d,
            size_current,
            arr_indexs_temp2__d,
            arr_vals_temp2__d);

        size_current /= 2;
        arr_indexs_current__d = arr_indexs_temp2__d;
        arr_vals_current__d = arr_vals_temp2__d;
    }
    if (recursion_depth % 2) {
        if (arr_vals_temp2__d[0] > arr_vals_temp2__d[1]) {
            *max_index__d = arr_indexs_temp2__d[0];
            *max_val__d = arr_vals_temp2__d[0];
        }
        else {
            *max_index__d = arr_indexs_temp2__d[1];
            *max_val__d = arr_vals_temp2__d[1];
        }
    }
    else {
        *max_index__d = arr_indexs_temp2__d[0];
        *max_val__d = arr_vals_temp2__d[0];
    }
};

[[deprecated]] __device__ size_t partition__d(
    int low,
    int up,
    size_t* arr_indexs__d,
    float* arr_vals__d)
{
    float pivot = arr_vals__d[up];
    int i = low - 1;
    float temp = 0;
    for (size_t j = low; j < up; ++j) {
        if (arr_vals__d[j] > pivot) {
            ++i;
            temp = arr_vals__d[i];
            arr_vals__d[i] = arr_vals__d[j];
            arr_vals__d[j] = temp;
            arr_indexs__d[i] = j;
            arr_indexs__d[j] = i;
        }
    }
    ++i;
    temp = arr_vals__d[i];
    arr_vals__d[i] = arr_vals__d[up];
    arr_vals__d[up] = temp;
    arr_indexs__d[i] = up;
    arr_indexs__d[up] = i;

    return i;
};

[[deprecated]] __global__ void quick_sort_no_limit__d(
    int low,
    int up,
    size_t* arr_indexs__d,
    float* arr_vals__d)
{
    if (low == up) return;

    // recursion
    int partition_index = partition__d(
        low,
        up,
        arr_indexs__d,
        arr_vals__d);
    if (partition_index > low + 1) {
        quick_sort_no_limit__d<<<1, 1>>>(
            low,
            partition_index - 1,
            arr_indexs__d,
            arr_vals__d);
    }
    if (partition_index + 1 < up) {
        quick_sort_no_limit__d<<<1, 1>>>(
            partition_index + 1,
            up,
            arr_indexs__d,
            arr_vals__d);
    }
};

[[deprecated]] __global__ void quick_sort_limited__d(
    int low,
    int up,
    size_t recursion_limit,
    size_t* arr_indexs__d,
    float* arr_vals__d)
{
    // base case
    if (up - low <= recursion_limit) {
        for (size_t i = low; i <= up; ++i) {
            for (size_t j = i + 1; j <= up; ++j) {
                if (arr_vals__d[j] > arr_vals__d[i]) {
                    float temp = arr_vals__d[i];
                    arr_vals__d[i] = arr_vals__d[j];
                    arr_vals__d[j] = temp;
                    arr_indexs__d[i] = j;
                    arr_indexs__d[j] = i;
                }
            }
        }
        return;
    }

    // recursion
    int partition_index = partition__d(
        low,
        up,
        arr_indexs__d,
        arr_vals__d);
    if (partition_index > low + 1) {
        quick_sort_limited__d<<<1, 1>>>(
            low,
            partition_index - 1,
            recursion_limit,
            arr_indexs__d,
            arr_vals__d);
    }
    if (partition_index + 1 < up) {
        quick_sort_limited__d<<<1, 1>>>(
            partition_index + 1,
            up,
            recursion_limit,
            arr_indexs__d,
            arr_vals__d);
    }
};

[[deprecated]] cudaError_t max_element_recursive__h(
    size_t size_,
    size_t recursion_limit_max_element,
    size_t stack_size_thread_req_max_element,
    size_t stack_size_thread_reset,
    float* arr_vals__d,
    size_t* max_index__d,
    float* max_val__d)
{
    cudaError_t cuda_status = cudaSuccess;

    // set the stack size
    cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, stack_size_thread_req_max_element);
    if (cuda_status != cudaSuccess) return cuda_status;

    // get max element
    if (!recursion_limit_max_element) {
        max_element_recursive_no_limit__d<<<1, 1>>>(
            arr_vals__d,
            0,
            size_ - 1,
            max_index__d,
            max_val__d);
    }
    else {
        max_element_recursive_limited__d<<<1, 1>>>(
            arr_vals__d,
            0,
            size_ - 1,
            recursion_limit_max_element,
            max_index__d,
            max_val__d);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    // reset the stack size
    cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, stack_size_thread_reset);
    return cuda_status;
};

cudaError_t max_element_no_recursion__h(
    size_t* arr_indexs__d,
    float* arr_vals__d,
    size_t size_,
    size_t* arr_indexs_temp1__d,
    size_t* arr_indexs_temp2__d,
    float* arr_vals_temp1__d,
    float* arr_vals_temp2__d,
    size_t* max_index__d,
    float* max_val__d)
{
    cudaError_t cuda_status = cudaSuccess;

    // initialize the index array
    dim3 grid_dim((size_ + BLOCK_DIM1) / BLOCK_DIM0);
    initialize_indexs<<<grid_dim, BLOCK_DIM0>>>(
        size_,
        arr_indexs__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    // determine the max
    max_element_no_recursion_main__d<<<1, 1>>>(
        arr_indexs__d,
        arr_vals__d,
        size_,
        arr_indexs_temp1__d,
        arr_indexs_temp2__d,
        arr_vals_temp1__d,
        arr_vals_temp2__d,
        max_index__d,
        max_val__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    return cuda_status;
};

[[deprecated]] cudaError_t quick_sort_recursive__h(
    size_t size_,
    size_t recursion_limit_quick_sort,
    size_t stack_size_thread_req_quick_sort,
    size_t stack_size_thread_reset,
    size_t* arr_indexs__d,
    float* arr_vals__d)
{
    cudaError_t cuda_status = cudaSuccess;

    // set the stack size
    cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, stack_size_thread_req_quick_sort);
    if (cuda_status != cudaSuccess) return cuda_status;

    // sort
    if (recursion_limit_quick_sort == 0) {
        quick_sort_no_limit__d<<<1, 1>>>(
            0,
            size_ - 1,
            arr_indexs__d,
            arr_vals__d);
    }
    else {
        quick_sort_limited__d<<<1, 1>>>(
            0,
            size_ - 1,
            recursion_limit_quick_sort,
            arr_indexs__d,
            arr_vals__d);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    // reset the stack size
    cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, stack_size_thread_reset);
    return cuda_status;
};

__global__ void sort_indexs_by_max_element_helper__d(
    size_t* arr_indexs__d,
    float* arr_vals__d,
    size_t* max_index__d,
    size_t i)
{
    size_t temp_index = arr_indexs__d[i];
    arr_indexs__d[i] = arr_indexs__d[*max_index__d];
    arr_indexs__d[*max_index__d] = temp_index;
    arr_vals__d[*max_index__d] = arr_vals__d[i];
    arr_vals__d[i] = FLT_MIN;
};

cudaError_t sort_indexs_by_max_element_no_recursion__h(
    size_t* arr_indexs__d,
    float* arr_vals__d,
    size_t size_,
    size_t sorted_element_count,
    size_t* max_index__d,
    float* max_val__d,
    size_t* arr_indexs_temp1__d,
    size_t* arr_indexs_temp2__d,
    float* arr_vals_temp1__d,
    float* arr_vals_temp2__d)
{
    cudaError_t cuda_status = cudaSuccess;
    for (size_t i = 0; i < sorted_element_count; ++i) {
        // determine the max
        max_element_no_recursion_main__d<<<1, 1>>>(
            arr_indexs__d,
            arr_vals__d,
            size_,
            arr_indexs_temp1__d,
            arr_indexs_temp2__d,
            arr_vals_temp1__d,
            arr_vals_temp2__d,
            max_index__d,
            max_val__d);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) return cuda_status;
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) return cuda_status;

        // move the max front and make the current max FLT_MIN
        sort_indexs_by_max_element_helper__d<<<1, 1>>>(
            arr_indexs__d,
            arr_vals__d,
            max_index__d,
            i);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) return cuda_status;
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) return cuda_status;
    }
    return cuda_status;
};

__global__ void create_stackings__d(
    size_t population_size,
    unsigned long long seed,
    _Stacking_t* stackings__d)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size) return;

    curandState state;
    curand_init(seed, tid, 0, &state);

    unsigned long long high = (unsigned long long)(curand(&state) & 0x7FFFFFFFFFFFFFFF);
    unsigned long long low = (unsigned long long)(curand(&state) & 0x7FFFFFFFFFFFFFFF);
    unsigned long long val = (high << 32) | low;
    stackings__d[tid] = val;
};

cudaError_t create_stackings__h(
    size_t population_size,
    _Stacking_t* stackings__d)
{
    cudaError_t cuda_status = cudaSuccess;

    dim3 grid_dim((population_size + BLOCK_DIM1) / BLOCK_DIM0);
    create_stackings__d<<<grid_dim, BLOCK_DIM0>>>(
        population_size,
        1234ULL,
        stackings__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

__device__ void get_stacking_angle_indexs__d(
    _Stacking_t stacking,
    _Stacking_angle_index_t* stacking_angle_indexs)
{
    size_t loc = PLY_COUNT;
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        --loc;
        stacking_angle_indexs[i] = (stacking >> (2 * loc)) & 0x3;
    }
};

__global__ void mutate_stackings__d(
    size_t population_size,
    unsigned long long seed,
    _Stacking_t* stackings__d)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size) return;

    // get new random angle index at a new random position
    curandState state;
    curand_init(seed, tid, 0, &state);

    // Generate a 64-bit random number and constrain it to [min, max]
    unsigned long long index_index_long = curand(&state) & 0x7FFFFFFFFFFFFFFF;
    unsigned long long index_val_long = curand(&state) & 0x7FFFFFFFFFFFFFFF;
    size_t index_index = (size_t)(index_index_long % (PLY_COUNT - 1));
    unsigned char index_val = static_cast<unsigned char>(index_val_long % (ANGLE_INDEX_90 + 1));

    // Clear the two old bits
    stackings__d[tid] &= ~(_Stacking_t(0x3) << index_index);

    // Set the new 2-bit value
    stackings__d[tid] |= index_val << index_index;
};

cudaError_t mutate_stackings__h(
    size_t population_size,
    _Stacking_t* stackings__d)
{
    cudaError_t cuda_status = cudaSuccess;

    dim3 grid_dim((population_size + BLOCK_DIM1) / BLOCK_DIM0);
    mutate_stackings__d<<<grid_dim, BLOCK_DIM0>>>(
        population_size,
        1234ULL,
        stackings__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

__global__ void filter_stackings__d(
    _Stacking_t const* stackings_crossover__d,
    size_t population_size_elite_use,
    size_t const* fitness_rate_indexs__d,
    _Stacking_t* stackings_elite__d)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size_elite_use) return;
    stackings_elite__d[tid] = stackings_crossover__d[fitness_rate_indexs__d[tid]];
};

[[deprecated]] cudaError_t get_stackings_elite_recursive__h(
    _Stacking_t const* stackings_crossover__d,
    size_t population_size_elite,
    size_t recursion_limit_max_element,
    size_t recursion_limit_quick_sort,
    size_t stack_size_thread_req_max_element,
    size_t stack_size_thread_req_quick_sort,
    size_t stack_size_thread_reset,
    float* fitness_rate_vals__d,
    size_t* fitness_rate_indexs__d,
    float* fitness_rate_sorted_vals__d,
    _Stacking_t* stackings_elite__d)
{
    cudaError_t cuda_status = cudaSuccess;

    // sizes
    size_t population_size_elite_use = population_size_elite + 1;
    size_t population_size_crossover = CROSSOVER_TYPE_COUNT * population_size_elite * population_size_elite;

    // initialize the index array
    dim3 grid_dim((population_size_crossover + BLOCK_DIM1) / BLOCK_DIM0);
    initialize_indexs<<<grid_dim, BLOCK_DIM0>>>(
        population_size_crossover,
        fitness_rate_indexs__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    // perform quick sort on the fitness rates
    cuda_status = quick_sort_recursive__h(
        population_size_crossover,
        recursion_limit_quick_sort,
        stack_size_thread_req_quick_sort,
        stack_size_thread_reset,
        fitness_rate_indexs__d,
        fitness_rate_vals__d);
    if (cuda_status != cudaSuccess) return cuda_status;

    // filter the crossover population using the sorted fitness rates
    filter_stackings__d<<<1, population_size_elite_use>>>(
        stackings_crossover__d,
        population_size_elite_use,
        fitness_rate_indexs__d,
        stackings_elite__d);
    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

cudaError_t get_stackings_elite__h(
    _Stacking_t const* stackings_crossover__d,
    size_t* fitness_rate_indexs__d,
    float* fitness_rate_vals__d,
    size_t population_size_elite,
    size_t* max_index__d,
    float* max_val__d,
    size_t* arr_indexs_temp1__d,
    size_t* arr_indexs_temp2__d,
    float* arr_vals_temp1__d,
    float* arr_vals_temp2__d,
    _Stacking_t* stackings_elite__d)
{
    cudaError_t cuda_status = cudaSuccess;

    // sizes
    size_t population_size_elite_use = population_size_elite + 1;
    size_t population_size_crossover = CROSSOVER_TYPE_COUNT * population_size_elite * population_size_elite;

    // initialize the index array
    dim3 grid_dim((population_size_crossover + BLOCK_DIM1) / BLOCK_DIM0);
    initialize_indexs<<<grid_dim, BLOCK_DIM0>>>(
        population_size_crossover,
        fitness_rate_indexs__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) return cuda_status;

    // sort for sorted_element_count
    cuda_status = sort_indexs_by_max_element_no_recursion__h(
        fitness_rate_indexs__d,
        fitness_rate_vals__d,
        population_size_crossover,
        population_size_elite_use,
        max_index__d,
        max_val__d,
        arr_indexs_temp1__d,
        arr_indexs_temp2__d,
        arr_vals_temp1__d,
        arr_vals_temp2__d);
    if (cuda_status != cudaSuccess) return cuda_status;

    // filter the crossover population using the sorted fitness rates
    filter_stackings__d<<<1, population_size_elite_use>>>(
        stackings_crossover__d,
        population_size_elite_use,
        fitness_rate_indexs__d,
        stackings_elite__d);
    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

__device__ _Stacking_t crossover_2__d(
    _Stacking_t stacking1,
    _Stacking_t stacking2)
{
    return (_Stacking_t)(
        (stacking1 & 0xFFFFFFFF00000000) |
        (stacking2 & 0x00000000FFFFFFFF));
};

__device__ _Stacking_t crossover_4__d(
    _Stacking_t stacking1,
    _Stacking_t stacking2)
{
    return (_Stacking_t)(
        (stacking1 & 0xFFFF0000FFFF0000) |
        (stacking2 & 0x0000FFFF0000FFFF));
};

__device__ _Stacking_t crossover_8__d(
    _Stacking_t stacking1,
    _Stacking_t stacking2)
{
    return (_Stacking_t)(
        (stacking1 & 0xFF00FF00FF00FF00) |
        (stacking2 & 0x00FF00FF00FF00FF));
};

__device__ _Stacking_t crossover_16__d(
    _Stacking_t stacking1,
    _Stacking_t stacking2)
{
    return (_Stacking_t)(
        (stacking1 & 0xF0F0F0F0F0F0F0F0) |
        (stacking2 & 0x0F0F0F0F0F0F0F0F));
};

__global__ void get_stackings_crossover__d(
    _Stacking_t const* stackings_elite__d,
    _Stacking_t* stackings_crossover__d)
{
    size_t i_elite1 = blockIdx.x;
    size_t i_elite2 = (i_elite1 + threadIdx.x + 1) % (blockDim.x + 1);
    size_t i_crossover = CROSSOVER_TYPE_COUNT * (i_elite1 * blockDim.x + i_elite2 - 1);
    stackings_crossover__d[i_crossover] = crossover_2__d(
        stackings_elite__d[i_elite1],
        stackings_elite__d[i_elite2]);
    stackings_crossover__d[i_crossover + 1] = crossover_4__d(
        stackings_elite__d[i_elite1],
        stackings_elite__d[i_elite2]);
    stackings_crossover__d[i_crossover + 2] = crossover_8__d(
        stackings_elite__d[i_elite1],
        stackings_elite__d[i_elite2]);
    stackings_crossover__d[i_crossover + 3] = crossover_16__d(
        stackings_elite__d[i_elite1],
        stackings_elite__d[i_elite2]);
};

cudaError_t get_stackings_crossover__h(
    _Stacking_t const* stackings_elite__d,
    size_t population_size_elite,
    _Stacking_t* stackings_crossover__d)
{
    cudaError_t cuda_status = cudaSuccess;

    get_stackings_crossover__d<<<population_size_elite, population_size_elite>>>(
        stackings_elite__d,
        stackings_crossover__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

__device__ void calculate_ABDs__d(
    _Stacking_t stacking,
    float* ABDs)
{
    _Stacking_angle_index_t stacking_angle_indexs[PLY_COUNT];
    get_stacking_angle_indexs__d(stacking, stacking_angle_indexs);

    size_t angle_index_counts[] = { 0, 0, 0, 0 };
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        ++angle_index_counts[stacking_angle_indexs[i]];
    }
    for (size_t i = 0; i < A_ARR_SIZE; ++i) {
        ABDs[i] = 0;
        for (size_t j = 0; j < ANGLE_INDEX_ARR_SIZE; ++j) {
            ABDs[i] += As_cache__d[i][j] * angle_index_counts[j];
        }
    }
    for (size_t i = A_ARR_SIZE; i < ABD_ARR_SIZE; ++i) {
        ABDs[i] = 0;
        for (size_t j = 0; j < PLY_COUNT; ++j) {
            ABDs[i] += Ds_cache__d[i - A_ARR_SIZE][j][stacking_angle_indexs[j]];
        }
    }
};

__device__ inline float calculate_fitness_rate_1__d(
    float fitness_coeff,
    float ABD_treshold,
    float ABD_val,
    float negative_fitness_factor)
{
    float fitness_val = fitness_coeff * (ABD_val - ABD_treshold) / ABD_treshold;
    if (fitness_val < 0) { fitness_val *= negative_fitness_factor; }
    return fitness_val;
};

__device__ inline float calculate_fitness_rate_2__d(
    float fitness_coeff,
    float ABD_val)
{
    return -abs(fitness_coeff * ABD_val);
};

__device__ inline float calculate_fitness_rate_main__d(
    float ABDs_current__d[ABD_ARR_SIZE],
    float negative_fitness_factor)
{
    return
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[0],
            ABDs_treshold__d[0],
            ABDs_current__d[0],
            negative_fitness_factor) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[1],
            ABDs_treshold__d[1],
            ABDs_current__d[1],
            negative_fitness_factor) +
        calculate_fitness_rate_2__d(
            fitness_coeffs__d[2],
            ABDs_current__d[2]) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[3],
            ABDs_treshold__d[3],
            ABDs_current__d[3],
            negative_fitness_factor) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[4],
            ABDs_treshold__d[4],
            ABDs_current__d[4],
            negative_fitness_factor) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[5],
            ABDs_treshold__d[5],
            ABDs_current__d[5],
            negative_fitness_factor) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[6],
            ABDs_treshold__d[6],
            ABDs_current__d[6],
            negative_fitness_factor) +
        calculate_fitness_rate_1__d(
            fitness_coeffs__d[7],
            ABDs_treshold__d[7],
            ABDs_current__d[7],
            negative_fitness_factor);
};

__global__ void calculate_fitness_rate__d(
    _Stacking_t const* stackings_crossover__d,
    size_t population_size_crossover,
    float negative_fitness_factor,
    float* fitness_rate_vals__d)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= population_size_crossover) return;

    float ABDs_current__d__d[ABD_ARR_SIZE];
    calculate_ABDs__d(stackings_crossover__d[tid], ABDs_current__d__d);
    fitness_rate_vals__d[tid] = calculate_fitness_rate_main__d(
        ABDs_current__d__d,
        negative_fitness_factor);
};

cudaError_t calculate_fitness_rates__h(
    _Stacking_t const* stackings_crossover__d,
    size_t population_size_crossover,
    float negative_fitness_factor,
    float* fitness_rate_vals__d)
{
    cudaError_t cuda_status = cudaSuccess;

    dim3 grid_dim((population_size_crossover + BLOCK_DIM1) / BLOCK_DIM0);
    calculate_fitness_rate__d<<<grid_dim, BLOCK_DIM0>>>(
        stackings_crossover__d,
        population_size_crossover,
        negative_fitness_factor,
        fitness_rate_vals__d);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaDeviceSynchronize();

    return cuda_status;
};

cudaError_t get_best_fit_stacking__h(
    _Stacking_t const* stackings_crossover__d,
    size_t population_size_crossover,
    size_t* best_fit_index__h,
    _Stacking_t* best_fit_stacking__h)
{
    cudaError_t cuda_status = cudaSuccess;

    _Stacking_t* stackings_crossover__h =
        (_Stacking_t*)malloc(population_size_crossover * sizeof(_Stacking_t));
    cuda_status = cudaMemcpy(
        stackings_crossover__h,
        stackings_crossover__d,
        population_size_crossover * sizeof(_Stacking_t),
        cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        free(stackings_crossover__h);
        return cuda_status;
    }
    *best_fit_stacking__h = stackings_crossover__h[*best_fit_index__h];
    free(stackings_crossover__h);

    return cuda_status;
};

cudaError_t run(
    Orthotropic const* orthotropic,
    size_t population_size_elite,
    float const fitness_coeffs__h[ABD_ARR_SIZE],
    size_t fail_count_max,
    size_t fail_count_restart,
    _Stacking_angle_index_t* best_fit_stacking__h_angle_indexs__h)
{
    cudaError_t cuda_status = cudaSuccess;

    // Parameters
    size_t population_size_elite_use{ population_size_elite + 1 };
    size_t population_size_crossover{ CROSSOVER_TYPE_COUNT * population_size_elite * population_size_elite };
    float negative_fitness_factor{ 16 };

    // cache arrays per angle index
    float cubic_difs[PLY_COUNT];
    generate_cubic_difs(orthotropic, PLY_COUNT, cubic_difs);
    float Qs[Q_ARR_SIZE];
    calculate_Qs(orthotropic, Qs);
    float Qbar_arr[ANGLE_INDEX_ARR_SIZE][QBAR_ARR_SIZE];
    calculate_Qbar_arr(Qs, Qbar_arr);

    float As_cache__h[A_ARR_SIZE][ANGLE_INDEX_ARR_SIZE];
    float Ds_cache__h[D_ARR_SIZE][PLY_COUNT][ANGLE_INDEX_ARR_SIZE];
    calculate_As_cache(orthotropic, Qbar_arr, As_cache__h);
    calculate_Ds_cache(Qbar_arr, cubic_difs, Ds_cache__h);

    float ABDs_treshold__h[ABD_ARR_SIZE];
    calculate_ABDs_treshold_main(
        As_cache__h,
        Ds_cache__h,
        ABDs_treshold__h);

    // set gpu
    cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) return cuda_status;

    // constants
    cuda_status = cudaMemcpyToSymbol(
        fitness_coeffs__d,
        fitness_coeffs__h,
        ABD_ARR_SIZE * sizeof(float));
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaMemcpyToSymbol(
        As_cache__d,
        As_cache__h,
        A_ARR_SIZE * ANGLE_INDEX_ARR_SIZE * sizeof(float));
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaMemcpyToSymbol(
        Ds_cache__d,
        Ds_cache__h,
        D_ARR_SIZE * PLY_COUNT * ANGLE_INDEX_ARR_SIZE * sizeof(float));
    if (cuda_status != cudaSuccess) return cuda_status;
    cuda_status = cudaMemcpyToSymbol(
        ABDs_treshold__d,
        ABDs_treshold__h,
        ABD_ARR_SIZE * sizeof(float));
    if (cuda_status != cudaSuccess) return cuda_status;
    // allocate the gpu buffers
    _Stacking_t* stackings_elite__d = 0;
    _Stacking_t* stackings_crossover__d = 0;
    size_t* fitness_rate_indexs__d = 0;
    float* fitness_rate_vals__d = 0;
    size_t* max_index__d = 0;
    float* max_val__d = 0;
    size_t* arr_indexs_temp1__d;
    size_t* arr_indexs_temp2__d;
    float* arr_vals_temp1__d;
    float* arr_vals_temp2__d;
    size_t* best_fit_index__d;
    float* best_fit_val__d;
    
    cuda_status = cudaMalloc(
        (void**)&stackings_elite__d,
        population_size_elite_use * sizeof(_Stacking_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&stackings_crossover__d,
        population_size_crossover * sizeof(_Stacking_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&fitness_rate_indexs__d,
        population_size_crossover * sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&fitness_rate_vals__d,
        population_size_crossover * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&max_index__d,
        sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&max_val__d,
        sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&arr_indexs_temp1__d,
        population_size_crossover / 2 * sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&arr_indexs_temp2__d,
        population_size_crossover / 4 * sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&arr_vals_temp1__d,
        population_size_crossover / 2 * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        cudaFree(arr_vals_temp1__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&arr_vals_temp2__d,
        population_size_crossover / 4 * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        cudaFree(arr_vals_temp1__d);
        cudaFree(arr_vals_temp2__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&best_fit_index__d,
        sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        cudaFree(arr_vals_temp1__d);
        cudaFree(arr_vals_temp2__d);
        cudaFree(best_fit_index__d);
        return cuda_status;
    }
    cuda_status = cudaMalloc(
        (void**)&best_fit_val__d,
        sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        cudaFree(arr_vals_temp1__d);
        cudaFree(arr_vals_temp2__d);
        cudaFree(best_fit_index__d);
        cudaFree(best_fit_val__d);
        return cuda_status;
    }

    // generate the initial elite population
    cuda_status = create_stackings__h(
        population_size_elite_use,
        stackings_elite__d);
    if (cuda_status != cudaSuccess) {
        cudaFree(stackings_elite__d);
        cudaFree(stackings_crossover__d);
        cudaFree(fitness_rate_indexs__d);
        cudaFree(fitness_rate_vals__d);
        cudaFree(max_index__d);
        cudaFree(max_val__d);
        cudaFree(arr_indexs_temp1__d);
        cudaFree(arr_indexs_temp2__d);
        cudaFree(arr_vals_temp1__d);
        cudaFree(arr_vals_temp2__d);
        cudaFree(best_fit_index__d);
        cudaFree(best_fit_val__d);
        return cuda_status;
    }

    // TODO: SIL
    clock_t t0 = clock();
    clock_t ti = t0;
    float durs{};

    // Run the main Loop
    size_t counter{};
    size_t fail_count_current = 0;
    size_t best_fit_index__h = 0;
    _Stacking_t best_fit_stacking__h = 0;
    float best_fit_rate_current__h = 0.f;
    float best_fit_rate_previous__h = 0.f;
    float best_fit_rate_max__h = 0.f;
    bool check_new_generation = true;
    while (fail_count_current < fail_count_max) {
        // mutate if not a new generation
        if (!check_new_generation) {
            cuda_status = mutate_stackings__h(
                population_size_elite_use,
                stackings_elite__d);
            if (cuda_status != cudaSuccess) {
                cudaFree(stackings_elite__d);
                cudaFree(stackings_crossover__d);
                cudaFree(fitness_rate_indexs__d);
                cudaFree(fitness_rate_vals__d);
                cudaFree(max_index__d);
                cudaFree(max_val__d);
                cudaFree(arr_indexs_temp1__d);
                cudaFree(arr_indexs_temp2__d);
                cudaFree(arr_vals_temp1__d);
                cudaFree(arr_vals_temp2__d);
                cudaFree(best_fit_index__d);
                cudaFree(best_fit_val__d);
                return cuda_status;
            }
        }

        // crossover
        cuda_status = get_stackings_crossover__h(
            stackings_elite__d,
            population_size_elite,
            stackings_crossover__d);
        if (cuda_status != cudaSuccess) {
            cudaFree(stackings_elite__d);
            cudaFree(stackings_crossover__d);
            cudaFree(fitness_rate_indexs__d);
            cudaFree(fitness_rate_vals__d);
            cudaFree(max_index__d);
            cudaFree(max_val__d);
            cudaFree(arr_indexs_temp1__d);
            cudaFree(arr_indexs_temp2__d);
            cudaFree(arr_vals_temp1__d);
            cudaFree(arr_vals_temp2__d);
            cudaFree(best_fit_index__d);
            cudaFree(best_fit_val__d);
            return cuda_status;
        }

        // calculate the fitness rates
        cuda_status = calculate_fitness_rates__h(
            stackings_crossover__d,
            population_size_crossover,
            negative_fitness_factor,
            fitness_rate_vals__d);
        if (cuda_status != cudaSuccess) {
            cudaFree(stackings_elite__d);
            cudaFree(stackings_crossover__d);
            cudaFree(fitness_rate_indexs__d);
            cudaFree(fitness_rate_vals__d);
            cudaFree(max_index__d);
            cudaFree(max_val__d);
            cudaFree(arr_indexs_temp1__d);
            cudaFree(arr_indexs_temp2__d);
            cudaFree(arr_vals_temp1__d);
            cudaFree(arr_vals_temp2__d);
            cudaFree(best_fit_index__d);
            cudaFree(best_fit_val__d);
            return cuda_status;
        }

        // the max fitness ratio
        cuda_status = max_element_no_recursion__h(
            fitness_rate_indexs__d,
            fitness_rate_vals__d,
            population_size_crossover,
            arr_indexs_temp1__d,
            arr_indexs_temp2__d,
            arr_vals_temp1__d,
            arr_vals_temp2__d,
            best_fit_index__d,
            best_fit_val__d);
        if (cuda_status != cudaSuccess) {
            cudaFree(stackings_elite__d);
            cudaFree(stackings_crossover__d);
            cudaFree(fitness_rate_indexs__d);
            cudaFree(fitness_rate_vals__d);
            cudaFree(max_index__d);
            cudaFree(max_val__d);
            cudaFree(arr_indexs_temp1__d);
            cudaFree(arr_indexs_temp2__d);
            cudaFree(arr_vals_temp1__d);
            cudaFree(arr_vals_temp2__d);
            cudaFree(best_fit_index__d);
            cudaFree(best_fit_val__d);
            return cuda_status;
        }
        cuda_status = cudaMemcpy(
            &best_fit_index__h,
            best_fit_index__d,
            sizeof(size_t),
            cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            cudaFree(stackings_elite__d);
            cudaFree(stackings_crossover__d);
            cudaFree(fitness_rate_indexs__d);
            cudaFree(fitness_rate_vals__d);
            cudaFree(max_index__d);
            cudaFree(max_val__d);
            cudaFree(arr_indexs_temp1__d);
            cudaFree(arr_indexs_temp2__d);
            cudaFree(arr_vals_temp1__d);
            cudaFree(arr_vals_temp2__d);
            cudaFree(best_fit_index__d);
            cudaFree(best_fit_val__d);
            return cuda_status;
        }
        cuda_status = cudaMemcpy(
            &best_fit_rate_current__h,
            best_fit_val__d,
            sizeof(float),
            cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            cudaFree(stackings_elite__d);
            cudaFree(stackings_crossover__d);
            cudaFree(fitness_rate_indexs__d);
            cudaFree(fitness_rate_vals__d);
            cudaFree(max_index__d);
            cudaFree(max_val__d);
            cudaFree(arr_indexs_temp1__d);
            cudaFree(arr_indexs_temp2__d);
            cudaFree(arr_vals_temp1__d);
            cudaFree(arr_vals_temp2__d);
            cudaFree(best_fit_index__d);
            cudaFree(best_fit_val__d);
            return cuda_status;
        }

        // inspect the current generation fitness values
        check_new_generation = false;
        if (best_fit_rate_current__h > best_fit_rate_max__h) {
            fail_count_current = 0;
            best_fit_rate_max__h = best_fit_rate_current__h;
            cuda_status = get_best_fit_stacking__h(
                stackings_crossover__d,
                population_size_crossover,
                &best_fit_index__h,
                &best_fit_stacking__h);
            if (cuda_status != cudaSuccess) {
                cudaFree(stackings_elite__d);
                cudaFree(stackings_crossover__d);
                cudaFree(fitness_rate_indexs__d);
                cudaFree(fitness_rate_vals__d);
                cudaFree(max_index__d);
                cudaFree(max_val__d);
                cudaFree(arr_indexs_temp1__d);
                cudaFree(arr_indexs_temp2__d);
                cudaFree(arr_vals_temp1__d);
                cudaFree(arr_vals_temp2__d);
                cudaFree(best_fit_index__d);
                cudaFree(best_fit_val__d);
                return cuda_status;
            }
        }
        else if (best_fit_rate_current__h <= best_fit_rate_previous__h) {
            ++fail_count_current;
            if (fail_count_current == fail_count_max) { break; }
            if (fail_count_current % fail_count_restart == 0) {
                check_new_generation = true;
                cuda_status = create_stackings__h(
                    population_size_elite_use,
                    stackings_elite__d);
                if (cuda_status != cudaSuccess) {
                    cudaFree(stackings_elite__d);
                    cudaFree(stackings_crossover__d);
                    cudaFree(fitness_rate_indexs__d);
                    cudaFree(fitness_rate_vals__d);
                    cudaFree(max_index__d);
                    cudaFree(max_val__d);
                    cudaFree(arr_indexs_temp1__d);
                    cudaFree(arr_indexs_temp2__d);
                    cudaFree(arr_vals_temp1__d);
                    cudaFree(arr_vals_temp2__d);
                    cudaFree(best_fit_index__d);
                    cudaFree(best_fit_val__d);
                    return cuda_status;
                }
            }
        }
        best_fit_rate_previous__h = best_fit_rate_current__h;

        // get the elite stackings
        if (!check_new_generation) {
            cuda_status = get_stackings_elite__h(
                stackings_crossover__d,
                fitness_rate_indexs__d,
                fitness_rate_vals__d,
                population_size_elite,
                max_index__d,
                max_val__d,
                arr_indexs_temp1__d,
                arr_indexs_temp2__d,
                arr_vals_temp1__d,
                arr_vals_temp2__d,
                stackings_elite__d);
            if (cuda_status != cudaSuccess) {
                cudaFree(stackings_elite__d);
                cudaFree(stackings_crossover__d);
                cudaFree(fitness_rate_indexs__d);
                cudaFree(fitness_rate_vals__d);
                cudaFree(max_index__d);
                cudaFree(max_val__d);
                cudaFree(arr_indexs_temp1__d);
                cudaFree(arr_indexs_temp2__d);
                cudaFree(arr_vals_temp1__d);
                cudaFree(arr_vals_temp2__d);
                cudaFree(best_fit_index__d);
                cudaFree(best_fit_val__d);
                return cuda_status;
            }
        }

        // TODO: SIL
        ++counter;
        clock_t t1 = clock();
        size_t dur = (t1 - t0) * 1000 / CLOCKS_PER_SEC;
        t0 = t1;
        durs += (float)dur;
        printf("**************************\n");
        printf("counter: %llu\n", counter);
        printf("duration: %llu\n", dur);
        printf("best_fit_index__h: %llu\n", best_fit_index__h);
        printf("best_fit_rate_current__h: %f\n", best_fit_rate_current__h);
        printf("best_fit_rate_previous__h: %f\n", best_fit_rate_previous__h);
        printf("best_fit_rate_max__h: %f\n", best_fit_rate_max__h);
    }

    // get the stacking angle indexs
    get_stacking_angle_indexs__h(best_fit_stacking__h, best_fit_stacking__h_angle_indexs__h);

    // free the gpu buffers
    cudaFree(stackings_elite__d);
    cudaFree(stackings_crossover__d);
    cudaFree(fitness_rate_indexs__d);
    cudaFree(fitness_rate_vals__d);
    cudaFree(max_index__d);
    cudaFree(max_val__d);
    cudaFree(arr_indexs_temp1__d);
    cudaFree(arr_indexs_temp2__d);
    cudaFree(arr_vals_temp1__d);
    cudaFree(arr_vals_temp2__d);
    cudaFree(best_fit_index__d);
    cudaFree(best_fit_val__d);



    // TODO: SIL
    clock_t tf = clock();
    size_t dur = (tf - ti) * 1000 / CLOCKS_PER_SEC;
    printf("\n**************************\n");
    printf("counter: %llu\n", counter);
    printf("total duration: %llu\n", dur);
    printf("average duration: %f\n", durs / (float)counter);
    printf("best_fit_rate_max__h: %f\n", best_fit_rate_max__h);



    return cuda_status;
};

int main()
{
    Orthotropic orthotropic;
    size_t population_size_elite = 64;
    float fitness_coeffs__h[ABD_ARR_SIZE] = { 40.f, 5.f, (float)pow(2, 20), 30.f, 10.f, 40.f, 20.f, 3.f };
    size_t fail_count_max = 100;
    size_t fail_count_restart = 10;
    _Stacking_angle_index_t* best_fit_stacking__h_angle_indexs__h;
    best_fit_stacking__h_angle_indexs__h = (_Stacking_angle_index_t*)malloc(PLY_COUNT * sizeof(_Stacking_angle_index_t));
    cudaError_t cuda_status = run(
        &orthotropic,
        population_size_elite,
        fitness_coeffs__h,
        fail_count_max,
        fail_count_restart,
        best_fit_stacking__h_angle_indexs__h);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "failed: %s\n", cudaGetErrorString(cuda_status));
        return 0;
    }
    for (size_t i = 0; i < PLY_COUNT; ++i) {
        printf("%d,", best_fit_stacking__h_angle_indexs__h[i]);
    }
    return 0;
}
