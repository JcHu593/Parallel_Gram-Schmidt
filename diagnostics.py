# save as diagnostics.py
import numpy as np
import sys

def read_vectors(path):
    with open(path) as f:
        n = int(f.readline().strip())
        A = np.loadtxt(f, max_rows=n)
    return A

V = read_vectors("testcases/n4096_s1_in.txt")   # adjust paths as needed
Q = read_vectors("output/n4096_s1_out.txt")

# Gram matrix (Q @ Q^T), check deviation from identity
G = Q @ Q.T
n = G.shape[0]
I = np.eye(n)
max_dev = np.max(np.abs(G - I))
diag_devs = np.abs(np.diag(G) - 1.0)
offdiag_max = np.max(np.abs(G - np.diag(np.diag(G))))
print("max |G - I| =", max_dev)
print("max |diag(G)-1| =", np.max(diag_devs))
print("max off-diag |G_ij| =", offdiag_max)

# Condition number of V
try:
    s = np.linalg.svd(V, compute_uv=False)
    cond = s[0] / s[-1]
    print("cond(V) =", cond)
except Exception as e:
    print("SVD failed:", e)