#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cuda.h>

#define BLOCK_SIZE 256
#define MAX_BLOCKS 1024

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

// Kernel to compute dot product of two vectors using reduction
// Uses a two-phase reduction: first within blocks, then across blocks
__global__ void dot_product_kernel(const double* a, const double* b, double* block_results, int n, int num_blocks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = num_blocks * blockDim.x;
    
    __shared__ double shared_sum[BLOCK_SIZE];
    double sum = 0.0;
    
    // Each thread accumulates its portion
    for (int i = tid; i < n; i += stride) {
        sum += a[i] * b[i];
    }
    
    shared_sum[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction within block using warp-level optimization
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no __syncthreads needed within a warp)
    if (threadIdx.x < 32) {
        volatile double* vshared = shared_sum;
        vshared[threadIdx.x] += vshared[threadIdx.x + 32];
        vshared[threadIdx.x] += vshared[threadIdx.x + 16];
        vshared[threadIdx.x] += vshared[threadIdx.x + 8];
        vshared[threadIdx.x] += vshared[threadIdx.x + 4];
        vshared[threadIdx.x] += vshared[threadIdx.x + 2];
        vshared[threadIdx.x] += vshared[threadIdx.x + 1];
    }
    
    // Write block result
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = shared_sum[0];
    }
}

// Kernel to reduce block results to a single value
__global__ void reduce_blocks_kernel(double* block_results, double* final_result, int num_blocks) {
    __shared__ double shared_sum[BLOCK_SIZE];
    
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

// Kernel to normalize a vector
__global__ void normalize_kernel(double* vec, double mag, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        vec[idx] /= mag;
    }
}

// Kernel to compute projection and subtract: v_j -= (dot(v_j, v_i)) * v_i
__global__ void orthogonalize_kernel(double* v_j, const double* v_i, double proj, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        v_j[idx] -= proj * v_i[idx];
    }
}

// Structure to hold pre-allocated GPU resources
struct GpuResources {
    double* d_block_results;
    double* d_final_result;
    int max_blocks;
    
    GpuResources(int n) {
        max_blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        CUDA_CHECK(cudaMalloc(&d_block_results, max_blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_final_result, sizeof(double)));
    }
    
    ~GpuResources() {
        cudaFree(d_block_results);
        cudaFree(d_final_result);
    }
};

double dot_product_gpu(const double* d_a, const double* d_b, int n, GpuResources& res) {
    int num_blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, res.max_blocks);
    
    dot_product_kernel<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, res.d_block_results, n, num_blocks);
    reduce_blocks_kernel<<<1, BLOCK_SIZE>>>(res.d_block_results, res.d_final_result, num_blocks);
    
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

    std::vector<std::vector<double>> vectors(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            infile >> vectors[i][j];
        }
    }
    infile.close();

    // Allocate device memory for vectors (stored as flat array)
    double* d_vectors;
    size_t vector_bytes = (size_t)n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_vectors, vector_bytes));
    
    // Copy vectors to device
    std::vector<double> flat_vectors(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flat_vectors[i * n + j] = vectors[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_vectors, flat_vectors.data(), vector_bytes, cudaMemcpyHostToDevice));

    // Pre-allocate GPU resources for dot product operations
    GpuResources gpuRes(n);
    
    std::chrono::steady_clock::time_point computation_start = std::chrono::steady_clock::now();

    int num_blocks = min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
    
    // Modified Gram-Schmidt on GPU
    for (int i = 0; i < n; ++i) {
        double* d_vec_i = d_vectors + i * n;
        
        // Compute magnitude of vector i
        double dot_i_i = dot_product_gpu(d_vec_i, d_vec_i, n, gpuRes);
        double mag = std::sqrt(dot_i_i);
        
        if (mag > 1e-9) {
            // Normalize vector i
            normalize_kernel<<<num_blocks, BLOCK_SIZE>>>(d_vec_i, mag, n);
        }
        
        // Orthogonalize remaining vectors
        for (int j = i + 1; j < n; ++j) {
            double* d_vec_j = d_vectors + j * n;
            
            // Compute projection: proj = dot(v_j, v_i)
            double proj = dot_product_gpu(d_vec_j, d_vec_i, n, gpuRes);
            
            // Subtract projection: v_j -= proj * v_i
            orthogonalize_kernel<<<num_blocks, BLOCK_SIZE>>>(d_vec_j, d_vec_i, proj, n);
        }
    }
    
    // Synchronize after all kernel launches
    CUDA_CHECK(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point computation_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = computation_end - computation_start;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(flat_vectors.data(), d_vectors, vector_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            vectors[i][j] = flat_vectors[i * n + j];
        }
    }

    CUDA_CHECK(cudaFree(d_vectors));

    // Write output
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return 1;
    }

    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            outfile << vectors[i][j] << (j == n - 1 ? "" : " ");
        }
        outfile << std::endl;
    }
    outfile.close();

    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
    
    std::cout << "Computation Time: " << elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Total Time: " << total_elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Compute-To-Total Time Ratio: " << (elapsed_seconds.count() / total_elapsed_seconds.count()) << std::endl;
    std::cout << "Output written to: " << output_path << std::endl;

    return 0;
}
