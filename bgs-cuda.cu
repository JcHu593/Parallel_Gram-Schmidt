#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 16  // Block size for the algorithm
#define MAX_REDUCE_BLOCKS 1024

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

// ----------------------------------------------------------------------------------
// Kernel 1: 計算 Block 與 Block 之間的相關性矩陣 M (Inter-Block)
// M[r][c] = dot(Block_j[r], Block_k[c])
// ----------------------------------------------------------------------------------
__global__ void compute_correlation_matrix_kernel(
    const double* A, double* M,
    int n, int j, int k, int current_bs, int k_bs
) {
    int r = blockIdx.y;  // M 的 row (對應 Block_j 的 row r)
    int c = blockIdx.x;  // M 的 col (對應 Block_k 的 row c)
    
    if (r >= current_bs || c >= k_bs) return;
    
    int row_curr = j + r;
    int row_prev = k + c;
    
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double sum = 0.0;
    
    // Grid-Stride Loop 計算 Dot Product
    for (int x = threadIdx.x; x < n; x += blockDim.x) {
        sum += A[row_curr * n + x] * A[row_prev * n + x];
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Block Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        // 寫入相關性矩陣 M (尺寸為 current_bs x k_bs)
        M[r * k_bs + c] = shared_sum[0];
    }
}

// ----------------------------------------------------------------------------------
// Kernel 2: 使用相關性矩陣更新 Block (Inter-Block Update)
// Block_j = Block_j - M * Block_k
// ----------------------------------------------------------------------------------
__global__ void update_block_kernel(
    double* A, const double* M,
    int n, int j, int k, int current_bs, int k_bs
) {
    int r = blockIdx.y;  // Block_j 的 row index
    if (r >= current_bs) return;
    
    int row_curr = j + r;
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index in matrix A
    
    if (x >= n) return;
    
    double update_val = 0.0;
    // 矩陣乘法: update += M[r, c] * Block_k[c, x]
    for (int c = 0; c < k_bs; ++c) {
        double scale = M[r * k_bs + c];
        int row_prev = k + c;
        update_val += scale * A[row_prev * n + x];
    }
    
    A[row_curr * n + x] -= update_val;
}

// ----------------------------------------------------------------------------------
// Kernel 3: Intra-Block Orthogonalization (關鍵優化!)
// 這個 Kernel 負責處理一個小 Block (例如 32xN 或 64xN) 的內部正交化。
// 為了避免 D2H，我們在這裡循序(Serial)處理 Block 內的每一行，但「行內計算」是平行的。
// ----------------------------------------------------------------------------------
__global__ void intra_block_mgs_kernel(double* A, int n, int j, int current_bs) {
    // 這裡每一個 Block 負責處理 "Block_j" 的全部正交化工作
    // 因為各行之間有相依性，必須由同一個 CTA 循序處理 i=0..bs-1
    // 為了效能，這個 Kernel 應該只啟動 1 個 Block (或者很少的 Blocks 配合 Global Memory 同步)
    // 簡單起見：啟動 <<<1, THREADS_PER_BLOCK>>>

    for (int i = 0; i < current_bs; ++i) {
        int row_i = j + i;

        // --- 1. 計算 Norm ---
        // 使用 Shared Memory 進行 Reduction
        __shared__ double s_norm_sq;
        if (threadIdx.x == 0) s_norm_sq = 0.0;
        __syncthreads();

        double local_sum = 0.0;
        for (int x = threadIdx.x; x < n; x += blockDim.x) {
            local_sum += A[row_i * n + x] * A[row_i * n + x];
        }

        // Warp Reduction for local_sum -> s_norm_sq
        // 簡單一點使用 atomicAdd 或者標準 reduction
        // 為了通用性這裡寫標準 reduction
        __shared__ double reduc_shmem[THREADS_PER_BLOCK];
        reduc_shmem[threadIdx.x] = local_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                reduc_shmem[threadIdx.x] += reduc_shmem[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        double norm_sq = reduc_shmem[0];
        double norm = sqrt(norm_sq);
        __shared__ double s_inv_norm;
        if (threadIdx.x == 0) s_inv_norm = (norm > 1e-9) ? 1.0 / norm : 0.0;
        __syncthreads();

        // --- 2. Normalize Row i ---
        double inv_norm_val = s_inv_norm;
        for (int x = threadIdx.x; x < n; x += blockDim.x) {
            A[row_i * n + x] *= inv_norm_val;
        }
        __syncthreads(); // 確保 Row i 更新完畢

        // --- 3. Orthogonalize Remaining Rows (k > i) ---
        for (int k = i + 1; k < current_bs; ++k) {
            int row_k = j + k;
            
            // 計算 Dot Product (row_k . row_i)
            double local_dot = 0.0;
            for (int x = threadIdx.x; x < n; x += blockDim.x) {
                local_dot += A[row_k * n + x] * A[row_i * n + x];
            }
            
            reduc_shmem[threadIdx.x] = local_dot;
            __syncthreads();

            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    reduc_shmem[threadIdx.x] += reduc_shmem[threadIdx.x + s];
                }
                __syncthreads();
            }
            
            double proj = reduc_shmem[0]; // 得到投影量
            
            // 更新 Row k: row_k -= proj * row_i
            for (int x = threadIdx.x; x < n; x += blockDim.x) {
                A[row_k * n + x] -= proj * A[row_i * n + x];
            }
            __syncthreads(); // 確保 Row k 更新完畢，進入下一輪
        }
    }
}

// ----------------------------------------------------------------------------------
// Main Host Code
// ----------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point total_start = std::chrono::steady_clock::now();
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream infile(input_path);
    if (!infile.is_open()) { std::cerr << "Error opening input file." << std::endl; return 1; }
    int n;
    infile >> n;
    std::vector<double> A(n * n);
    for (int i = 0; i < n * n; ++i) infile >> A[i];
    infile.close();

    std::chrono::steady_clock::time_point H2D_start = std::chrono::steady_clock::now();
    double* d_A;
    double* d_M; // Correlation Matrix Buffer
    size_t matrix_bytes = (size_t)n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_M, BLOCK_SIZE * BLOCK_SIZE * sizeof(double))); // 最大只需這麼大
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    std::chrono::steady_clock::time_point H2D_end = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point computation_start = std::chrono::steady_clock::now();

    int num_elem_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- Block Modified Gram-Schmidt ---
    for (int j = 0; j < n; j += BLOCK_SIZE) {
        int current_bs = std::min(BLOCK_SIZE, n - j);

        // 1. Inter-Block: 將當前 Block (j) 與所有之前的 Block (k) 正交化
        for (int k = 0; k < j; k += BLOCK_SIZE) {
            int k_bs = std::min(BLOCK_SIZE, n - k);

            // Compute M = Block_j * Block_k^T
            dim3 grid_corr(k_bs, current_bs);
            compute_correlation_matrix_kernel<<<grid_corr, THREADS_PER_BLOCK>>>(
                d_A, d_M, n, j, k, current_bs, k_bs
            );

            // Update Block_j -= M * Block_k
            dim3 grid_update(num_elem_blocks, current_bs);
            update_block_kernel<<<grid_update, THREADS_PER_BLOCK>>>(
                d_A, d_M, n, j, k, current_bs, k_bs
            );
        }

        // 2. Intra-Block: 在 GPU 上直接處理當前 Block 內部的 MGS
        // 注意：這裡只啟動 1 個 Block 來循序處理這 64 根向量的相依性
        intra_block_mgs_kernel<<<1, THREADS_PER_BLOCK>>>(d_A, n, j, current_bs);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point computation_end = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point D2H_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_M));
    std::chrono::steady_clock::time_point D2H_end = std::chrono::steady_clock::now();

    // Output
    std::ofstream outfile(output_path);
    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < n; ++c) outfile << A[i * n + c] << (c == n - 1 ? "" : " ");
        outfile << std::endl;
    }
    outfile.close();

    std::chrono::duration<double> comp_time = computation_end - computation_start;
    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_time = total_end - total_start;
    std::chrono::duration<double> h2d_time = H2D_end - H2D_start;
    std::chrono::duration<double> d2h_time = D2H_end - D2H_start;
    std::cout << "Algorithm: Optimized Block MGS (Full GPU)" << std::endl;
    std::cout << "Computation Time: " << comp_time.count() << "s" << std::endl;
    std::cout << "H2D Time: " << h2d_time.count() << "s" << std::endl;
    std::cout << "D2H Time: " << d2h_time.count() << "s" << std::endl;
    std::cout << "Total Time (including I/O): " << total_time.count() << "s" << std::endl;
    std::cout << "Ratio: " << comp_time.count() / total_time.count() << std::endl;
    return 0;
}