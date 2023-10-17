/** 
 *  @file   custom_preconditioner_base.hpp 
 *  @brief  Custom Eigen Preconditioner Base Class
 *  @author Kaixi Matteo Chen 
 *  @date   2023-10-12 
 ***********************************************/

#ifndef H_CUSTOM_PRECONDITIONER_HPP
#define H_CUSTOM_PRECONDITIONER_HPP

#include <iostream>
#include <Eigen/Core>

template <typename Scalar>
class CustomPreconditionerBase
{
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;

  public:
    typedef typename Matrix::StorageIndex StorageIndex;
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic
    };

    CustomPreconditionerBase() : m_isInitialized(false) {
    }

    template<typename MatType>
    explicit CustomPreconditionerBase(const MatType& mat) : m_preconditioner_matrix(mat.rows(), mat.cols()) {
      compute(mat);
    }

    constexpr Eigen::Index rows() const EIGEN_NOEXCEPT {
      return m_preconditioner_matrix.rows();
    }
    constexpr Eigen::Index cols() const EIGEN_NOEXCEPT {
      return m_preconditioner_matrix.cols();
    }

    template<typename MatType>
    CustomPreconditionerBase& analyzePattern(const MatType& ) {
      return *this;
    }

    template<typename MatType>
    CustomPreconditionerBase& factorize(const MatType& mat) {
      m_preconditioner_matrix.resize(mat.rows(), mat.cols());
      m_preconditioner_matrix.setZero();
      for (int i=0; i<m_preconditioner_matrix.rows(); i++) {
        m_preconditioner_matrix(i,i) = 1;
      }
      m_isInitialized = true;
      return *this;
    }

    template<typename MatType>
    CustomPreconditionerBase& compute(const MatType& mat) {
      return factorize(mat);
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const {
      x = m_preconditioner_matrix * b;
    }

    template<typename Rhs> inline const Eigen::Solve<CustomPreconditionerBase, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const {
      eigen_assert(m_isInitialized && "CustomPreconditionerBase is not initialized.");
      /*
       * m_preconditioner_matrix is a nXn matrix multiplied for a nX1 vector
       * */
      eigen_assert(m_preconditioner_matrix.cols()==b.rows()
                && "CustomPreconditionerBase::solve(): invalid number of rows of the right hand side matrix b");
      return Eigen::Solve<CustomPreconditionerBase, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() {
      return Eigen::Success;
    };

    void update_initialiazed(const bool initialized) {
      m_isInitialized = initialized;
    }

    bool get_initialiazed() const {
      return m_isInitialized;
    }

  protected:
    bool m_isInitialized;
  
  private:
    Matrix m_preconditioner_matrix;
};

#endif //H_CUSTOM_PRECONDITIONER_HPP
