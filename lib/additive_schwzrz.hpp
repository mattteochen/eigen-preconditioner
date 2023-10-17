#ifndef H_ADDITIVE_SCHWARZ_HPP
#define H_ADDITIVE_SCHWARZ_HPP

#include <iostream>
#include <Eigen/Dense>
#include <utility>
#include <vector>
#include "custom_preconditioner_base.hpp"

#define DEBUG 0

template<typename Scalar, typename OriginalMatrix>
class AdditiveSchwarz : public CustomPreconditionerBase<Scalar>
{
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> SubMatrix;
    using Matrix = SubMatrix;
    static constexpr uint8_t CORES = 2;
  public:

    AdditiveSchwarz() : CustomPreconditionerBase<Scalar>() {
      std::cout << "You have chosen the AdditiveSchwarz based on the identity matrix" << std::endl;
    }

    template<typename MatType>
    explicit AdditiveSchwarz(const MatType& mat) : CustomPreconditionerBase<Scalar>(mat) {
      std::cout << "You have chosen the AdditiveSchwarz based on the identity matrix" << std::endl;
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

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
      //b is the kith residual
      //x is the result after applying the preconditioner on the residual
      x.resize(m_original_mat_size,1);
      for (uint32_t i=0; i<CORES; i++) {
        auto rhs = m_restrictions[i].transpose() * m_restrictions[i] * b;
        auto lhs = m_restrictions[i].transpose() * m_submatrixes[i] * m_restrictions[i];
        x += lhs.fullPivLu().solve(rhs);
      } 
    }

    template<typename Rhs> inline const Eigen::Solve<AdditiveSchwarz, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const {
      eigen_assert(CustomPreconditionerBase<Scalar>::get_initialiazed() && "AdditiveSchwarz is not initialized.");
      return Eigen::Solve<AdditiveSchwarz, Rhs>(*this, b.derived());
    }

  protected:
    std::vector<Matrix> m_restrictions;
    std::vector<SubMatrix> m_submatrixes;
    uint32_t m_original_mat_size;

  private:

    uint32_t get_subdomain_size(const int n) {
      return n/2+1;
    } 

    template<typename MatType>
    void initialize(const MatType& mat) {
      const uint32_t mat_size = mat.rows();
      m_original_mat_size = mat_size;
      const uint32_t size = get_subdomain_size(mat_size);
      const uint32_t q = mat_size - size;

      auto fill_subdomains = [&]() -> void {
        for (uint32_t k=0; k<m_submatrixes.size(); k++) {
          auto& m = m_submatrixes[k];
          m.resize(size, size);
          const uint32_t addition = k * q;
          for (uint32_t i=addition; i<addition+size; i++) {
            for (uint32_t j=addition; j<addition+size; j++) {
              m(i-addition,j-addition) = mat.coeff(i,j);
            }
          }
          #if(DEBUG==1)
          std::cout << "submatrix " << k << std::endl << m << std::endl; 
          #endif        
        }
      };
      auto fill_restrictions = [&]() -> void {
        for (uint32_t k=0; k<m_restrictions.size(); k++) {
          auto& m = m_restrictions[k];
          m.resize(size, mat_size);
          const uint32_t start_col = k * q;
          for (uint32_t i=0; i<size; i++) {
            m(i,start_col+i) = 1.0;
          }
          #if(DEBUG==1)
          std::cout << "restriction " << k << std::endl << m << std::endl; 
          #endif        
        }
      };
      m_restrictions.resize(CORES);
      m_submatrixes.resize(CORES);
      // m_original_matrix = mat;
      fill_subdomains();
      fill_restrictions();
      CustomPreconditionerBase<Scalar>::update_initialiazed(true);
    }
};

#endif //H_ADDITIVE_SCHWARZ_HPP
