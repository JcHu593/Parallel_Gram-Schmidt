## Gram-Schmidt
- function-like definition:
$f:\{\text{any\ set\ of\ }n\text{\ vectors\ in\ }\mathbb{R}^{n}\}\rightarrow \{\text{orthonormal\ basis\ of\ the\ subspace\ spanned\ by\ the\ input\ vectors}\}$
- Input format:
    - .txt file
    - first line: one integer (n) representing number of vectors involved
    - following n lines: each line represinting a vector of n double
    - set of the n vectors can be either linearly dependent or independent
- Output format:
    - same as input
- Answer checker :`./testcases/check_gs.py`
    - usage:<br>
      `python check_gs.py <INPUTFILE> <OUTPUTFILE>`
- Implementaion: e.g.`gs-seq.cc`
    - compilation:<br>
      `g++ [options] gs-seq.cc -o gs-seq`
    - execution:<br>
      `./gs-seq <INPUTFILE> <OUTPUTFILE>` 