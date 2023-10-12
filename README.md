# eigen-preconditioner
Custom matrix-based preconditioner base class for Eigen iterative solvers.

## How to build a preconditioner from the base class
Any non trivial preconditioner should override (inheriting the base class) the `factorize` method.

## How to run the example test?
1. Be sure to have the Eigen library installed:
    - [HPC-Polimi](https://github.com/HPC-Courses/AMSC-Labs/tree/main/Labs/2023-24/lab00-setup)
    or
    - [Official](https://eigen.tuxfamily.org/index.php?title=Main_Page)
2. Set your Eigen library path in `compile_flags.txt`
3. Add `chmod` permissions to `run.sh`
4. `./run.sh`

### `compile_flags.txt`
This is the [clang](https://clangd.llvm.org/) language server configuration file to resolve external library imports. Can be used with any text editor that supports the clang language server (Vim, VSCode, ...).
