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
      `g++ [options] gs-seq.cc -o gs-seq`
    - execution:<br>
      `./gs-seq <INPUTFILE> <OUTPUTFILE>` 
- Answer checker :`./output/check_gs.py`
    - usage:<br>
      `python check_gs.py <INPUTFILE> <OUTPUTFILE>`