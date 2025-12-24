#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ----------------------------------------------------------------------
// Kernel 1: 單一向量的 Dot Product (用於計算自身長度/Norm)
// 每個 Block 計算一部分，最後由 CPU 或另一個 Kernel 匯總? 
// 為了效率，我們這裡使用由單一 Block 完成的 Reduction (假設 N < 1024*1024 夠用)
// 如果 N 非常大，需要多級 Reduction，但此處針對 N=4096 優化
// ----------------------------------------------------------------------
__global__ void single_vector_norm_squared_kernel(const double* vec, double* result, int n) {
    __shared__ double cache[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    double temp_sum = 0.0;
    
    // Grid-Stride Loop: 處理 N > BLOCK_SIZE 的情況
    for (int i = tid; i < n; i += blockDim.x) {
        temp_sum += vec[i] * vec[i];
    }
    
    cache[tid] = temp_sum;
    __syncthreads();
    
    // Block 內 Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) *result = cache[0];
}

// ----------------------------------------------------------------------
// Kernel 2: 正規化 (Normalize)
// 讀取 GPU 上的 norm_squared 結果，計算 sqrt 並除以它
// ----------------------------------------------------------------------
__global__ void normalize_vector_kernel(double* vec, const double* norm_sq_ptr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 讀取長度並開根號 (只有第一個 Thread 需要讀，但為了簡單讓大家讀取到 Shared 或是直接讀 Global)
    // 考慮到讀取次數極少，直接讀 Global 即可
    double mag = sqrt(*norm_sq_ptr);
    
    // 避免除以 0
    if (mag < 1e-9) return; 

    if (idx < n) {
        vec[idx] /= mag;
    }
}

// ----------------------------------------------------------------------
// Kernel 3: 批次計算投影量 (Batch Dot Product)
// Grid.x 代表剩下的向量數量 (j = i+1 ... n-1)
// 每個 Block 負責計算一個 dot(v_j, v_i)
// ----------------------------------------------------------------------
__global__ void batch_dot_product_kernel(const double* all_vectors, double* projections, 
                                         int current_i, int n, int num_remaining) {
    // blockIdx.x 對應到 "第幾個剩下的向量"
    int j_offset = blockIdx.x; 
    if (j_offset >= num_remaining) return;

    int target_j_index = current_i + 1 + j_offset; // 實際的向量索引 j

    const double* v_i = all_vectors + current_i * n;      // 基準向量
    const double* v_j = all_vectors + target_j_index * n; // 目標向量

    __shared__ double cache[BLOCK_SIZE];
    int tid = threadIdx.x;
    double temp_sum = 0.0;

    // 計算 dot(v_j, v_i)
    for (int k = tid; k < n; k += blockDim.x) {
        temp_sum += v_j[k] * v_i[k];
    }

    cache[tid] = temp_sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    // 寫入結果到 projections 陣列
    if (tid == 0) {
        projections[j_offset] = cache[0];
    }
}

// ----------------------------------------------------------------------
// Kernel 4: 批次更新向量 (Batch Update)
// v_j = v_j - proj * v_i
// Grid 2D: X=向量元素分塊, Y=剩下的向量數量
// ----------------------------------------------------------------------
__global__ void batch_update_kernel(double* all_vectors, const double* projections, 
                                    int current_i, int n, int num_remaining) {
    
    int j_offset = blockIdx.y; // 第幾個剩下的向量
    if (j_offset >= num_remaining) return;

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // 向量內的元素索引
    if (idx >= n) return;

    int target_j_index = current_i + 1 + j_offset;
    
    double proj = projections[j_offset]; // 讀取剛才算好的投影量
    
    double* v_i = all_vectors + current_i * n;
    double* v_j = all_vectors + target_j_index * n;

    v_j[idx] -= proj * v_i[idx];
}

// ----------------------------------------------------------------------
// Main Host Code
// ----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point total_start = std::chrono::steady_clock::now();

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    // 1. Read Input
    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file." << std::endl;
        return 1;
    }

    int n;
    infile >> n;
    
    // 使用一維陣列儲存矩陣 (Row-Major: 每個向量 v_i 連續儲存)
    std::vector<double> h_vectors(n * n);
    for (int i = 0; i < n * n; ++i) {
        infile >> h_vectors[i];
    }
    infile.close();

    // 2. Allocate GPU Memory
    double *d_vectors, *d_projections, *d_temp_norm;
    size_t vec_size = n * n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_vectors, vec_size));
    CUDA_CHECK(cudaMalloc(&d_projections, n * sizeof(double))); // 暫存投影係數
    CUDA_CHECK(cudaMalloc(&d_temp_norm, sizeof(double)));       // 暫存 Norm

    // 3. H2D Copy
    std::chrono::steady_clock::time_point H2D_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors.data(), vec_size, cudaMemcpyHostToDevice));
    std::chrono::steady_clock::time_point H2D_end = std::chrono::steady_clock::now();

    // 4. MGS Loop on GPU
    std::chrono::steady_clock::time_point comp_start = std::chrono::steady_clock::now();

    for (int i = 0; i < n; ++i) {
        // --- Step A: Normalize v_i ---
        // 1. 計算 v_i . v_i
        single_vector_norm_squared_kernel<<<1, BLOCK_SIZE>>>(d_vectors + i * n, d_temp_norm, n);
        
        // 2. 正規化 v_i (讀取 d_temp_norm 並更新 d_vectors)
        // 使用足夠的 Blocks 來覆蓋長度 n
        int num_blocks_norm = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        normalize_vector_kernel<<<num_blocks_norm, BLOCK_SIZE>>>(d_vectors + i * n, d_temp_norm, n);

        // --- Step B: Orthogonalize rest (Batch Processing) ---
        int num_remaining = n - (i + 1);
        
        if (num_remaining > 0) {
            // 3. 一次計算所有 dot(v_j, v_i) for j > i
            // Grid 大小 = 剩下的向量數，每個 Block 算一個向量
            batch_dot_product_kernel<<<num_remaining, BLOCK_SIZE>>>(d_vectors, d_projections, i, n, num_remaining);

            // 4. 一次更新所有 v_j = v_j - proj * v_i
            // Grid X = 向量長度分塊, Grid Y = 向量數量
            dim3 grid_update((n + BLOCK_SIZE - 1) / BLOCK_SIZE, num_remaining);
            batch_update_kernel<<<grid_update, BLOCK_SIZE>>>(d_vectors, d_projections, i, n, num_remaining);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point comp_end = std::chrono::steady_clock::now();

    // 5. D2H Copy
    std::chrono::steady_clock::time_point D2H_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(h_vectors.data(), d_vectors, vec_size, cudaMemcpyDeviceToHost));
    std::chrono::steady_clock::time_point D2H_end = std::chrono::steady_clock::now();

    // 6. Output to File
    std::ofstream outfile(output_path);
    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6);
    
    // 注意：原本的 vector<vector> 轉成了一維，輸出時要控制換行
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            outfile << h_vectors[i * n + j] << (j == n - 1 ? "" : " ");
        }
        outfile << std::endl;
    }
    outfile.close();

    // Cleanup
    cudaFree(d_vectors);
    cudaFree(d_projections);
    cudaFree(d_temp_norm);

    // Timing Report
    std::chrono::duration<double> total_time = std::chrono::steady_clock::now() - total_start;
    std::chrono::duration<double> comp_time = comp_end - comp_start;
    std::chrono::duration<double> h2d_time = H2D_end - H2D_start;
    std::chrono::duration<double> d2h_time = D2H_end - D2H_start;

    std::cout << "Computation Time: " << comp_time.count() << "s" << std::endl;
    std::cout << "H2D Time: " << h2d_time.count() << "s" << std::endl;
    std::cout << "D2H Time: " << d2h_time.count() << "s" << std::endl;
    std::cout << "Total Time: " << total_time.count() << "s" << std::endl;
    std::cout << "Ratio: " << comp_time.count() / total_time.count() << std::endl;

    return 0;
}