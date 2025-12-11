#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n> <seed>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    int base_seed = std::stoi(argv[2]);

    std::string filename = "n" + std::to_string(n) + "_s" + std::to_string(base_seed) + "_in.txt";
    
    // Buffer to store generated values to allow parallel generation without I/O races
    std::vector<double> data(n * n);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            unsigned int element_seed = base_seed + i * n + j;
            
            // Initialize generator with the specific seed
            std::minstd_rand gen(element_seed);
            std::uniform_real_distribution<double> dis(-10.0, 10.0);
            
            data[i * n + j] = dis(gen);
        }
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    outfile << n << "\n";
    //outfile << std::fixed << std::setprecision(6); // Set precision for output
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            outfile << data[i * n + j] << (j == n - 1 ? "" : " ");
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Generated " << filename << std::endl;

    return 0;
}
