/* 2.87s */
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <omp.h>

// Function to calculate dot product of two vectors
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Function to calculate magnitude of a vector
double magnitude(const std::vector<double>& a) {
    return std::sqrt(dot_product(a, a));
}

// Modified Gram-Schmidt process
void gram_schmidt(std::vector<std::vector<double>>& vectors, int n) {
    for (int i = 0; i < n; ++i) {
        // Normalize the current vector
        double mag = magnitude(vectors[i]);
        if (mag > 1e-9) { // Avoid division by zero
            #pragma omp simd
            for (int k = 0; k < n; ++k) {
                vectors[i][k] /= mag;
            }
        }
    
        // Orthogonalize the remaining vectors against the current one
        #pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            double proj = dot_product(vectors[j], vectors[i]);
            #pragma omp simd
            for (int k = 0; k < n; ++k) {
                vectors[j][k] -= proj * vectors[i][k];
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

    std::vector<std::vector<double>> vectors(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            infile >> vectors[i][j];
        }
    }
    infile.close();

    std::chrono::steady_clock::time_point computation_start = std::chrono::steady_clock::now();
    gram_schmidt(vectors, n);
    std::chrono::steady_clock::time_point computation_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = computation_end - computation_start;
    
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening output file: " << output_path << std::endl;
        return 1;
    }

    outfile << n << std::endl;
    outfile << std::fixed << std::setprecision(6); // Set precision for output
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
