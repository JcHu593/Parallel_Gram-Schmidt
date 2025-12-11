## Gram-Schmidt
- function-like definition:
$f:\{\text{any\ set\ of\ }n\text{\ vectors\ in\ }\mathbb{R}^{n}\}\rightarrow \{\text{orthonormal\ basis\ of\ the\ subspace\ spanned\ by\ the\ input\ vectors}\}$
- Input format:
    - .txt file
    - first line: one **integer** (**1 <= n <= 4096**) representing number of vectors involved
    - following n lines: each line represinting a vector of n **doubles** ranging **[-10.0, 10.0]**
    - set of the n vectors can be either linearly dependent or independent
- Output format:
    - same as input
- Testcase generator:`./testcases/gen_testcase.cpp`
    - compilation:<br>
      `g++ gen_testcase.cpp -o gen_testcase -fopenmp`
    - execution:<br>
      `./gen_testcase <n> <seed>`
- Implementaion: e.g.`gs-seq.cc`
    - compilation:<br>
      `g++ -O3 -fopenmp -march=native [other options] gs-seq.cc -o gs-seq`
    - execution:<br>
      `./gs-seq <INPUTFILE> <OUTPUTFILE>` 
- Answer checker :`./output/check_gs.py`
    - usage:<br>
      `python check_gs.py <INPUTFILE> <OUTPUTFILE>`
    - result intepretation:<br>
      - `Orthonormality check PASSED.` -> The non-zero vectors do form an orthonormal (enough) basis.
      - `Span preservation check PASSED.` -> The spanned space is consistent (enough).

## Optimization
1) Sequential Modified Gram-Schmidt `gs-seq.cc`
2) Multi-threaded Modified Gram-Schmidt `gs-mthreads_v1.cc`
3) Vectorized Multi-threaded Modified Gram-Schmidt `gs-mthreads_v1_v2.cc`
4) Vectorized Multi-threaded Blocked Modified Gram-Schmidt `gs-bmgs.cc`