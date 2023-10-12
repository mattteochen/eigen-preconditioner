/** 
 *  @file   custom_preconditioner_base.cpp 
 *  @brief  Custom Eigen Preconditioner Base Class Exaple Usage
 *  @author Kaixi Matteo Chen 
 *  @date   2023-10-12 
 ***********************************************/

#include "custom_preconditioner_base.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <iostream>
#include <string>

constexpr unsigned MATRIX_SIZE = 100;
/*
 * tridiagonal matrix setting
 * */
constexpr unsigned RESERVE_SIZE = MATRIX_SIZE + (2*MATRIX_SIZE-1);
constexpr double TOLLERANCE = 1.e-10;
constexpr unsigned MAX_ITERATIONS = 1000;

using std::cout;
using std::endl;
using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

auto fill_matrix = [](SpMat& m) -> void {
  for (int i=0; i<MATRIX_SIZE; i++) {
    m.coeffRef(i, i) = 2.0*(i+1);
    if(i>0) m.coeffRef(i, i-1) = -i;
    if(i<MATRIX_SIZE-1) m.coeffRef(i, i+1) = -(i+1);
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
  cout << "This simple program compares Eigen built it DiagonalPreconditioner class with CustomPreconditionerBase class" << endl << endl;

  SpMat A(MATRIX_SIZE, MATRIX_SIZE);
  A.reserve(RESERVE_SIZE);
  fill_matrix(A);
  SpVec e = SpVec::Ones(A.rows());
  SpVec b = A * e;
  SpVec x(A.rows());

  test<SpMat, SpVec, SpVec, SpVec, CustomPreconditionerBase<double>>("#####Eigen native CG with custom preconditioner#####", A, b, x, e);
  x *= 0;
  test<SpMat, SpVec, SpVec, SpVec, Eigen::DiagonalPreconditioner<double>>("#####Eigen native CG with diagonal preconditioner#####", A, b, x, e);

  return 0;
}
