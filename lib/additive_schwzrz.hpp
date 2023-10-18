#ifndef H_ADDITIVE_SCHWARZ_HPP
#define H_ADDITIVE_SCHWARZ_HPP

/** 
 *  @file   additive_schwarz.hpp 
 *  @brief  Additive Schwarz Preconditioner Base Class
 *  @author Kaixi Matteo Chen 
 *  @date   2023-10-12 
 ***********************************************/

#include "custom_preconditioner_base.hpp"
#include "assertion.hpp"
#include "CTPL/ctpl_stl.h"
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#define MULTICORE 1
#define USE_LDLT 1

template<typename Scalar, typename OriginalMatrix>
class AdditiveSchwarz : public CustomPreconditionerBase<Scalar>
{
    //Current implementation will work only on tri-diagonal matrixes, submatrixes are not sparse as their dimension is 3x3
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> SubMatrix;
    using Matrix = SubMatrix;

    static constexpr uint8_t CORES = 1;
  public:

    AdditiveSchwarz() : CustomPreconditionerBase<Scalar>() {
    }

    template<typename MatType>
    explicit AdditiveSchwarz(const MatType& mat) : CustomPreconditionerBase<Scalar>(mat) {
      CustomPreconditionerBase<Scalar>::compute(mat);
    }

    template<typename MatType>
    AdditiveSchwarz& compute(const MatType& mat) {
      return factorize(mat);
    }

    template<typename MatType>
    AdditiveSchwarz& factorize(const MatType& mat) {
      initialize(mat);
      return *this;
    }

  #if (MULTICORE == 1)
    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
      ctpl::thread_pool t_pool(CORES);
      std::mutex mtx;
      auto callable = [&mtx, &x, &b, this](const uint32_t i) {
        // std::cout << "Thread " << i << std::endl;
        auto lhs = this->m_restrictions_t[i] * this->m_restrictions[i] * b;
        auto rhs = this->m_restrictions_t[i] * this->m_submatrixes[i] * this->m_restrictions[i];
        std::unique_lock<std::mutex> lock(mtx);
      #if (USE_LDLT == 1)
        Eigen::LDLT<Matrix> solver_LDLT(rhs);
        // if not positive or negative semidefinite change the solver class
        if (solver_LDLT.info() == Eigen::NumericalIssue) {
          Eigen::HouseholderQR<Matrix> solver_HQR(rhs);
          x += solver_HQR.solve(lhs);
        } else {
          x += solver_LDLT.solve(lhs);
        }
      #else
        Eigen::HouseholderQR<Matrix> solver(rhs);
        x += solver->solve(lhs);
      #endif
      };

      x.resize(m_original_mat_size,1);
      std::vector<std::future<void>> results(CORES);
      for (uint32_t sub=0; sub<m_num_submatrixes;) {
        for (uint32_t i=0; i<CORES; i++) {
          results[i] = t_pool.push(
              [&mtx, &x, &b, this, sub](uint32_t) {
              // std::cout << "Thread " << sub << std::endl;
              auto lhs = this->m_restrictions_t[sub] * this->m_restrictions[sub] * b;
              auto rhs = this->m_restrictions_t[sub] * this->m_submatrixes[sub] * this->m_restrictions[sub];
            #if (USE_LDLT == 1)
              Eigen::LDLT<Matrix> solver_LDLT(rhs);
              // if not positive or negative semidefinite change the solver class
              if (solver_LDLT.info() == Eigen::NumericalIssue) {
                Eigen::HouseholderQR<Matrix> solver_HQR(rhs);
                std::unique_lock<std::mutex> lock(mtx);
                x += solver_HQR.solve(lhs);
              } else {
                std::unique_lock<std::mutex> lock(mtx);
                x += solver_LDLT.solve(lhs);
              }
            #else
              Eigen::HouseholderQR<Matrix> solver(rhs);
              std::unique_lock<std::mutex> lock(mtx);
              x += solver->solve(lhs);
            #endif
            }
          );
          sub++;
        }
        for (uint32_t i=0; i<CORES; i++) {
          results[i].get();
        }
      }
    }
  #else
    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
      //b is the kith residual
      //x is the result after applying the preconditioner on the residual
      x.resize(m_original_mat_size,1);
      for (uint32_t i=0; i<CORES; i++) {
        auto lhs = m_restrictions_t[i] * m_restrictions[i] * b;
        auto rhs = m_restrictions_t[i] * m_submatrixes[i] * m_restrictions[i];
        x += rhs.fullPivLu().solve(lhs);
      } 
    }
  #endif

    template<typename Rhs> inline const Eigen::Solve<AdditiveSchwarz, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const {
      eigen_assert(CustomPreconditionerBase<Scalar>::get_initialiazed() && "AdditiveSchwarz is not initialized.");
      return Eigen::Solve<AdditiveSchwarz, Rhs>(*this, b.derived());
    }

  protected:
    static constexpr uint8_t SUBMATRIX_SIZE = 3;
    std::vector<Matrix> m_restrictions;
    std::vector<Matrix> m_restrictions_t;
    std::vector<SubMatrix> m_submatrixes;
    uint32_t m_num_submatrixes;
    uint32_t m_original_mat_size;

  private:
    
    //add concept for accepting non sparse matrix
    template<typename MatType>
    bool validate_tridiagonal(const MatType m) {
      for (uint32_t i=0; i<m.rows(); i++) {
        for (uint32_t j=0; j<m.cols(); j++) {
          if (i == j || (i && i-1 == j) || (j && j-1 == i)) {
            continue;
          }
          if (m.coeff(i, j) != 0) {
            return false;
          }
        }
      }
      return true;
    }

    template<typename MatType>
    void initialize(const MatType& mat) {
      ASSERT(mat.rows() == mat.cols(), "Input matrix is not square"); //can we remove this?
      ASSERT(validate_tridiagonal(mat), "Input matrix is not tridiagonal");
      m_original_mat_size = mat.rows();
      m_num_submatrixes = m_original_mat_size/2; //only for tridiagonal
      
      auto compute_last_subdomain_even_original_matrix_dimension = [&](const uint32_t k) {
        return k == m_submatrixes.size() - 1 && m_original_mat_size%2 == 0;
      };

      auto fill_subdomains = [&]() -> void {
        for (uint32_t k=0; k<m_submatrixes.size(); k++) {
          auto& m = m_submatrixes[k];
          m.resize(SUBMATRIX_SIZE, SUBMATRIX_SIZE);
          const uint32_t addition = k * (SUBMATRIX_SIZE-1); //overlap of one element of the domain
          std::pair<uint8_t, uint8_t> sub_matrix_indexes = {0,0};
          for (uint32_t i=addition; i<addition+SUBMATRIX_SIZE; i++, sub_matrix_indexes.first++) {
            sub_matrix_indexes.second = 0;
            for (uint32_t j=addition;  j<addition+SUBMATRIX_SIZE; j++, sub_matrix_indexes.second++) {
              uint32_t mat_i = i, mat_j = j;
              if (compute_last_subdomain_even_original_matrix_dimension(k)) {
                mat_i -= 1;
                mat_j -= 1;
              }
              m(sub_matrix_indexes.first, sub_matrix_indexes.second) = mat.coeff(mat_i, mat_j);
            }
          }
        }
      };
      auto fill_restrictions = [&]() -> void {
        for (uint32_t k=0; k<m_restrictions.size(); k++) {
          auto& m = m_restrictions[k];
          m.resize(SUBMATRIX_SIZE, m_original_mat_size);
          const uint32_t start_col = k * (SUBMATRIX_SIZE-1); //overlap of one element of the domain;
          for (uint32_t i=0; i<SUBMATRIX_SIZE; i++) {
            if (!compute_last_subdomain_even_original_matrix_dimension(k)) {
              m(i,start_col+i) = 1.0;
            } else {
              m(i,start_col+i-1) = 1.0;
            }
          }
        }
      };
      auto fill_restriction_transpose = [&]() {
        for (uint32_t i=0; i<m_num_submatrixes; ++i) {
          m_restrictions_t[i] = m_restrictions[i].transpose();
        }
      };
      m_restrictions.resize(m_num_submatrixes);
      m_restrictions_t.resize(m_num_submatrixes);
      m_submatrixes.resize(m_num_submatrixes);
      fill_subdomains();
      fill_restrictions();
      fill_restriction_transpose();
      CustomPreconditionerBase<Scalar>::update_initialiazed(true);
    }
};

#endif //H_ADDITIVE_SCHWARZ_HPP
