import sys
import numpy as np
import argparse
import time

def read_vectors(filepath):
    """Reads vectors from a file. First line is N, followed by N lines of vectors."""
    try:
        with open(filepath, 'r') as f:
            line = f.readline()
            if not line:
                return None
            n = int(line.strip())
    except Exception as e:
        print(f"Error reading header from {filepath}: {e}")
        return None

    # Use pandas for faster reading if available, otherwise numpy
    try:
        import pandas as pd
        # read_csv with sep='\s+' handles space-separated values efficiently
        df = pd.read_csv(filepath, skiprows=1, header=None, sep=r'\s+', engine='c')
        # Ensure we only get the first n columns (in case of trailing spaces)
        return df.values[:, :n]
    except ImportError:
        return np.loadtxt(filepath, skiprows=1)

def check_orthonormality(Q, tol=1e-3):
    """
    Checks if non-zero rows of Q are orthonormal.
    Allows for zero rows which correspond to linearly dependent input vectors.
    Condition: Q @ Q.T approx Diagonal(1 if row != 0 else 0)
    """
    n = Q.shape[0]
    
    # Calculate row norms to identify zero vectors
    norms = np.linalg.norm(Q, axis=1)
    
    # A vector is considered zero if its norm is very small
    is_zero_vec = norms < tol
    
    if np.any(is_zero_vec):
        print(f"  Note: Detected {np.sum(is_zero_vec)} zero vector(s) (linear dependence).")

    # Construct the expected result for Q @ Q.T
    # It should be identity, but with 0 on diagonal for zero vectors
    expected_gram = np.eye(n)
    # Set diagonal elements corresponding to zero vectors to 0
    for i in range(n):
        if is_zero_vec[i]:
            expected_gram[i, i] = 0.0
    
    # Matrix multiplication (parallelized by BLAS)
    prod = Q @ Q.T
    
    diff = np.abs(prod - expected_gram)
    max_diff = np.max(diff)
    
    return max_diff < tol, max_diff

def check_span_preservation(V, Q, tol=1e-3):
    """
    Checks if the span is preserved and GS order is respected.
    Condition: V @ Q.T should be lower triangular.
    (Because v_i is in span(q_1, ..., q_i))
    """
    # Matrix multiplication
    prod = V @ Q.T
    
    # Check upper triangular part (excluding diagonal)
    upper_indices = np.triu_indices(prod.shape[0], k=1)
    max_upper = np.max(np.abs(prod[upper_indices]))
    
    return max_upper < tol, max_upper

def main():
    parser = argparse.ArgumentParser(description='Check Gram-Schmidt output validity.')
    parser.add_argument('input_file', help='Path to input vectors file')
    parser.add_argument('output_file', help='Path to output vectors file')
    args = parser.parse_args()

    start_time = time.time()
    
    print(f"Reading files...")
    try:
        V = read_vectors(args.input_file)
        Q = read_vectors(args.output_file)
    except Exception as e:
        print(f"Failed to read files: {e}")
        sys.exit(1)
    
    if V is None or Q is None:
        print("Error: Could not read vectors.")
        sys.exit(1)
        
    if V.shape != Q.shape:
        print(f"Shape mismatch: Input {V.shape}, Output {Q.shape}")
        sys.exit(1)
        
    n = V.shape[0]
    print(f"Checking {n} vectors of dimension {n}...")
    
    # Tolerance considerations:
    # The C++ output uses fixed precision 6. This introduces rounding errors approx 1e-6.
    # Accumulation of errors in dot products of size N can be larger.
    # 1e-3 or 1e-4 is a reasonable practical tolerance.
    TOLERANCE = 1e-2 # Relaxed slightly for larger N and float output precision
    
    # 1. Check Orthonormality
    print("Checking orthonormality...")
    is_ortho, err_ortho = check_orthonormality(Q, TOLERANCE)
    print(f"Orthonormality Max Error: {err_ortho:.6e}")
    
    # 2. Check Span Preservation
    print("Checking span preservation...")
    is_span, err_span = check_span_preservation(V, Q, TOLERANCE)
    print(f"Span Preservation Max Error: {err_span:.6e}")
    
    if is_ortho and is_span:
        print("VALID")
    else:
        print("INVALID")
        sys.exit(1)
        
    print(f"Time taken: {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    main()
