/** 
 *  @file   main.cpp 
 *  @brief  Eigen Additive Schwarz Preconditioner Class Exaple Usage
 *  @author Kaixi Matteo Chen 
 *  @date   2023-10-12 
 ***********************************************/

#include "../../lib/additive_schwarz.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include<Eigen/SparseCholesky>
#include <iostream>
#include <string>
#include <chrono>

/*
 * still not parallel, do not go over 100
 * */
constexpr unsigned MATRIX_SIZE = 60;
/*
 * tridiagonal matrix setting
 * */
constexpr unsigned RESERVE_SIZE = MATRIX_SIZE + (2*MATRIX_SIZE-1);
constexpr double TOLLERANCE = 1.e-10;
constexpr unsigned MAX_ITERATIONS = 500;

using std::cout;
using std::endl;
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

auto fill_matrix = [](SpMat& m) -> void {
  for (int i=0; i<MATRIX_SIZE; i++) {
    m.coeffRef(i, i) = 2.0;
    if(i>0) m.coeffRef(i, i-1) = -1.0;
    if(i<MATRIX_SIZE-1) m.coeffRef(i, i+1) = -1.0;
  }
  cout << "Matrix A size:" << m.rows() << "X" << m.cols() << " Non zero entries:" << m.nonZeros() << endl;
};

template<typename Matrix, typename Unknown, typename Result, typename Error, typename Preconditioner>
void test(const std::string info, Matrix& A, Result& b, Unknown& x, Error& e) {
  Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, Preconditioner> solver;
  solver.setMaxIterations(MAX_ITERATIONS);
  solver.setTolerance(TOLLERANCE);
  solver.compute(A);
  x = solver.solve(b);
  std::cout << info << endl;
  std::cout << "#iterations:     " << solver.iterations() << endl;
  std::cout << "estimated error: " << solver.error() << endl;
  std::cout << "effective error: " << (x-e).norm() << endl;
  std::cout << "info flag:       " << solver.info() << endl;
}

int main() {
  SpMat A(MATRIX_SIZE, MATRIX_SIZE);
  A.reserve(RESERVE_SIZE);
  fill_matrix(A);
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A * e;
  SpVec x(A.rows());

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  test<SpMat, SpVec, SpVec, SpVec, AdditiveSchwarz<double, SpMat>>("#####Eigen native CG with custom preconditioner#####", A, b, x, e);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "CG with Additive Schwarz preconditioner computation time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
  x *= 0;
  begin = std::chrono::steady_clock::now();
  test<SpMat, SpVec, SpVec, SpVec, Eigen::DiagonalPreconditioner<double>>("#####Eigen native CG with diagonal preconditioner#####", A, b, x, e);
  end = std::chrono::steady_clock::now();
  std::cout << "CG with Jacobi preconditioner computation time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

  x *= 0;
  begin = std::chrono::steady_clock::now();
  Eigen::SimplicialLLT<SpMat> sparse_LU_solver(A);
  sparse_LU_solver.compute(A);
  if (sparse_LU_solver.info() != Eigen::Success) {
    std::cout << "Can not factorize A with Eigen::SimplicialLLT<>" << std::endl;
  }
  x = sparse_LU_solver.solve(b);
  end = std::chrono::steady_clock::now();
  std::cout << "Sparse LU direct method computation time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

  return 0;
}
