#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cuda.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 64  // Block size for the algorithm (same as CPU version)
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

// Kernel to compute a single element of the correlation matrix M
// M[r, c] = dot(Block_j[r], Block_k[c]) = dot(A[j+r], A[k+c])
__global__ void compute_correlation_matrix_kernel(
    const double* A, double* M,
    int n, int j, int k, int current_bs, int k_bs
) {
    int r = blockIdx.y;  // row in M (index in current block)
    int c = blockIdx.x;  // col in M (index in previous block)
    
    if (r >= current_bs || c >= k_bs) return;
    
    int row_curr = j + r;
    int row_prev = k + c;
    
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double sum = 0.0;
    
    // Each thread accumulates part of the dot product
    for (int x = threadIdx.x; x < n; x += blockDim.x) {
        sum += A[row_curr * n + x] * A[row_prev * n + x];
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction within block
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    if (threadIdx.x < 32) {
        volatile double* vshared = shared_sum;
        vshared[threadIdx.x] += vshared[threadIdx.x + 32];
        vshared[threadIdx.x] += vshared[threadIdx.x + 16];
        vshared[threadIdx.x] += vshared[threadIdx.x + 8];
        vshared[threadIdx.x] += vshared[threadIdx.x + 4];
        vshared[threadIdx.x] += vshared[threadIdx.x + 2];
        vshared[threadIdx.x] += vshared[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        M[r * k_bs + c] = shared_sum[0];
    }
}

// Kernel to update Block_j: Block_j = Block_j - M * Block_k
// Each block handles one row of Block_j
__global__ void update_block_kernel(
    double* A, const double* M,
    int n, int j, int k, int current_bs, int k_bs
) {
    int r = blockIdx.y;  // row index in current block
    if (r >= current_bs) return;
    
    int row_curr = j + r;
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index in matrix A
    
    if (x >= n) return;
    
    double update = 0.0;
    // Sum over all columns in M (i.e., all rows in Block_k)
    for (int c = 0; c < k_bs; ++c) {
        double scale = M[r * k_bs + c];
        int row_prev = k + c;
        update += scale * A[row_prev * n + x];
    }
    
    A[row_curr * n + x] -= update;
}

// Kernel to compute dot product of a row with itself (for magnitude)
__global__ void dot_product_self_kernel(
    const double* A, double* block_results,
    int n, int row, int num_blocks_used
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = num_blocks_used * blockDim.x;
    
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double sum = 0.0;
    
    for (int x = tid; x < n; x += stride) {
        double val = A[row * n + x];
        sum += val * val;
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x < 32) {
        volatile double* vshared = shared_sum;
        vshared[threadIdx.x] += vshared[threadIdx.x + 32];
        vshared[threadIdx.x] += vshared[threadIdx.x + 16];
        vshared[threadIdx.x] += vshared[threadIdx.x + 8];
        vshared[threadIdx.x] += vshared[threadIdx.x + 4];
        vshared[threadIdx.x] += vshared[threadIdx.x + 2];
        vshared[threadIdx.x] += vshared[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared_sum[0];
    }
}

// Kernel to reduce block results
__global__ void reduce_kernel(double* block_results, double* final_result, int num_blocks) {
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    
    double sum = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += block_results[i];
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x < 32) {
        volatile double* vshared = shared_sum;
        vshared[threadIdx.x] += vshared[threadIdx.x + 32];
        vshared[threadIdx.x] += vshared[threadIdx.x + 16];
        vshared[threadIdx.x] += vshared[threadIdx.x + 8];
        vshared[threadIdx.x] += vshared[threadIdx.x + 4];
        vshared[threadIdx.x] += vshared[threadIdx.x + 2];
        vshared[threadIdx.x] += vshared[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        *final_result = shared_sum[0];
    }
}

// Kernel to normalize a row
__global__ void normalize_row_kernel(double* A, int n, int row, double inv_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        A[row * n + x] *= inv_mag;
    }
}

// Kernel to zero out a row
__global__ void zero_row_kernel(double* A, int n, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        A[row * n + x] = 0.0;
    }
}

// Kernel to compute dot product between two rows
__global__ void dot_product_kernel(
    const double* A, double* block_results,
    int n, int row_a, int row_b, int num_blocks_used
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = num_blocks_used * blockDim.x;
    
    __shared__ double shared_sum[THREADS_PER_BLOCK];
    double sum = 0.0;
    
    for (int x = tid; x < n; x += stride) {
        sum += A[row_a * n + x] * A[row_b * n + x];
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x < 32) {
        volatile double* vshared = shared_sum;
        vshared[threadIdx.x] += vshared[threadIdx.x + 32];
        vshared[threadIdx.x] += vshared[threadIdx.x + 16];
        vshared[threadIdx.x] += vshared[threadIdx.x + 8];
        vshared[threadIdx.x] += vshared[threadIdx.x + 4];
        vshared[threadIdx.x] += vshared[threadIdx.x + 2];
        vshared[threadIdx.x] += vshared[threadIdx.x + 1];
    }
    
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared_sum[0];
    }
}

// Kernel to subtract projection: row_a = row_a - proj * row_b
__global__ void subtract_projection_kernel(double* A, int n, int row_a, int row_b, double proj) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n) {
        A[row_a * n + x] -= proj * A[row_b * n + x];
    }
}

// Structure to hold pre-allocated GPU resources
struct GpuResources {
    double* d_block_results;
    double* d_final_result;
    double* d_M;  // Correlation matrix (max size: BLOCK_SIZE x BLOCK_SIZE)
    int max_reduce_blocks;
    
    GpuResources(int n) {
        max_reduce_blocks = std::min((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, MAX_REDUCE_BLOCKS);
        CUDA_CHECK(cudaMalloc(&d_block_results, max_reduce_blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_M, BLOCK_SIZE * BLOCK_SIZE * sizeof(double)));
    }
    
    ~GpuResources() {
        cudaFree(d_block_results);
        cudaFree(d_final_result);
        cudaFree(d_M);
    }
};

// Helper function to compute dot product on GPU
double dot_product_gpu(const double* d_A, int n, int row_a, int row_b, GpuResources& res) {
    int num_blocks = std::min((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, res.max_reduce_blocks);
    
    if (row_a == row_b) {
        dot_product_self_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, res.d_block_results, n, row_a, num_blocks);
    } else {
        dot_product_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_A, res.d_block_results, n, row_a, row_b, num_blocks);
    }
    reduce_kernel<<<1, THREADS_PER_BLOCK>>>(res.d_block_results, res.d_final_result, num_blocks);
    
    double result;
    CUDA_CHECK(cudaMemcpy(&result, res.d_final_result, sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}

int main(int argc, char* argv[]) {
    std::chrono::steady_clock::time_point total_start = std::chrono::steady_clock::now();
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream infile(input_path);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file: " << input_path << std::endl;
        return 1;
    }

    int n;
    if (!(infile >> n)) {
        std::cerr << "Error reading number of vectors." << std::endl;
        return 1;
    }

    // Use a flat vector for better cache locality (Row-Major)
    std::vector<double> A(n * n);
    for (int i = 0; i < n * n; ++i) {
        infile >> A[i];
    }
    infile.close();

    // Allocate device memory
    double* d_A;
    size_t matrix_bytes = (size_t)n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_A, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), matrix_bytes, cudaMemcpyHostToDevice));

    // Pre-allocate GPU resources
    GpuResources gpuRes(n);

    std::chrono::steady_clock::time_point computation_start = std::chrono::steady_clock::now();

    int num_elem_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Block Modified Gram-Schmidt on GPU
    for (int j = 0; j < n; j += BLOCK_SIZE) {
        int current_bs = std::min(BLOCK_SIZE, n - j);

        // 1. Inter-Block Orthogonalization:
        // Orthogonalize current block 'j' against all previous blocks 'k'
        for (int k = 0; k < j; k += BLOCK_SIZE) {
            int k_bs = std::min(BLOCK_SIZE, n - k);

            // Compute correlation matrix M = Block_j * (Block_k)^T
            dim3 grid_corr(k_bs, current_bs);
            compute_correlation_matrix_kernel<<<grid_corr, THREADS_PER_BLOCK>>>(
                d_A, gpuRes.d_M, n, j, k, current_bs, k_bs
            );

            // Update Block_j: Block_j = Block_j - M * Block_k
            dim3 grid_update(num_elem_blocks, current_bs);
            update_block_kernel<<<grid_update, THREADS_PER_BLOCK>>>(
                d_A, gpuRes.d_M, n, j, k, current_bs, k_bs
            );
        }

        // 2. Intra-Block Orthogonalization:
        // Orthogonalize vectors within the current block using standard MGS
        for (int i = 0; i < current_bs; ++i) {
            int row_i = j + i;
            
            // Compute magnitude
            double mag_sq = dot_product_gpu(d_A, n, row_i, row_i, gpuRes);
            double mag = std::sqrt(mag_sq);

            if (mag > 1e-9) {
                // Normalize vector
                normalize_row_kernel<<<num_elem_blocks, THREADS_PER_BLOCK>>>(d_A, n, row_i, 1.0 / mag);
            } else {
                // Zero out linearly dependent vector
                zero_row_kernel<<<num_elem_blocks, THREADS_PER_BLOCK>>>(d_A, n, row_i);
            }

            // Orthogonalize remaining vectors in block against this vector
            for (int next = i + 1; next < current_bs; ++next) {
                int row_next = j + next;
                
                // Compute projection
                double proj = dot_product_gpu(d_A, n, row_next, row_i, gpuRes);
                
                // Subtract projection
                subtract_projection_kernel<<<num_elem_blocks, THREADS_PER_BLOCK>>>(d_A, n, row_next, row_i, proj);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point computation_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = computation_end - computation_start;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, matrix_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

    // Write output
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return 1;
    }

    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j_col = 0; j_col < n; ++j_col) {
            outfile << A[i * n + j_col] << (j_col == n - 1 ? "" : " ");
        }
        outfile << std::endl;
    }
    outfile.close();

    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
    
    std::cout << "Algorithm: Block Modified Gram-Schmidt (CUDA)" << std::endl;
    std::cout << "Computation Time: " << elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Total Time: " << total_elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Compute-To-Total Time Ratio: " << (elapsed_seconds.count() / total_elapsed_seconds.count()) << std::endl;
    std::cout << "Output written to: " << output_path << std::endl;

    return 0;
}
