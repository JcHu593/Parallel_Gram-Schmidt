#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <omp.h>

// Block size for the algorithm. 
// Tuning this affects cache performance. 32-128 is usually good for L1/L2.
#define BLOCK_SIZE 64 

// Helper: Dot product of two rows (vectors)
double dot_product(const std::vector<double>& flat_matrix, int n, int row_a, int row_b) {
    double sum = 0.0;
    // Vectorize this loop
    #pragma omp simd reduction(+:sum)
    for (int k = 0; k < n; ++k) {
        sum += flat_matrix[row_a * n + k] * flat_matrix[row_b * n + k];
    }
    return sum;
}

// Helper: Scale a row
void scale_row(std::vector<double>& flat_matrix, int n, int row, double factor) {
    #pragma omp simd
    for (int k = 0; k < n; ++k) {
        flat_matrix[row * n + k] *= factor;
    }
}

// Helper: Subtract scaled row_b from row_a (a = a - scale * b)
void saxpy_row(std::vector<double>& flat_matrix, int n, int row_a, int row_b, double scale) {
    #pragma omp simd
    for (int k = 0; k < n; ++k) {
        flat_matrix[row_a * n + k] -= scale * flat_matrix[row_b * n + k];
    }
}

// Block Modified Gram-Schmidt
void block_gram_schmidt(std::vector<double>& A, int n) {
    // A is stored in row-major order: A[i, j] = A[i * n + j]
    
    // Loop over blocks
    for (int j = 0; j < n; j += BLOCK_SIZE) {
        // Current block size (may be smaller at the end)
        int current_bs = std::min(BLOCK_SIZE, n - j); // Handle last block edge case
        
        // 1. Inter-Block Orthogonalization:
        // Orthogonalize current block 'j' against all previous blocks 'k'
        // Block 'k' range: [k, k + k_bs)
        // Block 'j' range: [j, j + current_bs)
        #pragma unroll
        for (int k = 0; k < j; k += BLOCK_SIZE) {
            int k_bs = std::min(BLOCK_SIZE, n - k);
            
            // Compute correlation matrix M = Block_j * (Block_k)^T
            // M dimensions: current_bs x k_bs
            std::vector<double> M(current_bs * k_bs, 0.0);

            // Compute M = Block_j * (Block_k)^T
            #pragma omp parallel for collapse(2) schedule(static)
            for (int r = 0; r < current_bs; ++r) {
                for (int c = 0; c < k_bs; ++c) {
                    double val = 0.0;
                    int row_curr = j + r;
                    int row_prev = k + c;
                    // Standard dot product between two rows
                    #pragma omp simd reduction(+:val)
                    for (int x = 0; x < n; ++x) {
                        val += A[row_curr * n + x] * A[row_prev * n + x];
                    }
                    M[r * k_bs + c] = val;
                }
            }

            // Update Block_j: Block_j = Block_j - M * Block_k
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < current_bs; ++r) {
                for (int c = 0; c < k_bs; ++c) {
                    double scale = M[r * k_bs + c];
                    int row_curr = j + r;
                    int row_prev = k + c;
                    
                    // Subtract scaled previous vector from current vector
                    // We can't easily vector-subtract efficiently across shared memory without 
                    // causing race conditions if we parallelize the inner reduction differently.
                    // Here, each thread owns 'r' (a row in Block_j), so it's safe.
                    #pragma omp simd
                    for (int x = 0; x < n; ++x) {
                        A[row_curr * n + x] -= scale * A[row_prev * n + x];
                    }
                }
            }
        }

        // 2. Intra-Block Orthogonalization:
        // Orthogonalize the vectors within the current block using standard MGS
        // Since the block is small (fits in cache), we might process sequentially or with fine-grained parallelism.
        // For simplicity and stability, we do this part sequentially for the outer loop, but parallelize the vector ops.
        #pragma unroll
        for (int i = 0; i < current_bs; ++i) {
            int row_i = j + i;
            double mag = std::sqrt(dot_product(A, n, row_i, row_i));
            
            if (mag > 1e-9) {
                scale_row(A, n, row_i, 1.0 / mag);
            } else {
                // Handle linearly dependent vectors (zero out)
                scale_row(A, n, row_i, 0.0);
            }

            // Parallelize the update of the remaining vectors in this block
            #pragma omp parallel for schedule(static)
            for (int next = i + 1; next < current_bs; ++next) {
                int row_next = j + next;
                double proj = dot_product(A, n, row_next, row_i);
                saxpy_row(A, n, row_next, row_i, proj);
            }
        }
    }
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

    std::chrono::steady_clock::time_point computation_start = std::chrono::steady_clock::now();
    
    block_gram_schmidt(A, n);

    std::chrono::steady_clock::time_point computation_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = computation_end - computation_start;
    
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return 1;
    }

    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            outfile << A[i * n + j] << (j == n - 1 ? "" : " ");
        }
        outfile << std::endl;
    }
    outfile.close();
    std::chrono::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed_seconds = total_end - total_start;
    
    std::cout << "Algorithm: Block Modified Gram-Schmidt" << std::endl;
    std::cout << "Computation Time: " << elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Total Time: " << total_elapsed_seconds.count() << "s" << std::endl;
    std::cout << "Compute-To-Total Time Ratio: " << (elapsed_seconds.count() / total_elapsed_seconds.count()) << std::endl;
    std::cout << "Output written to: " << output_path << std::endl;
    return 0;
}