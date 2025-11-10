#pragma once

#include <cstddef>      // std::size_t
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <qpOASES.hpp>

// Imprime matriz em buffer row-wise (A tem rows*cols elementos)
void print_matriz_qpoases(const qpOASES::real_t* A,
                          std::size_t rows,
                          std::size_t cols);

// Converte Sparse -> buffer DENSO row-wise (malloc). Use std::free depois.
qpOASES::real_t* to_rowwise_dense(const Eigen::SparseMatrix<double>& B);

// Converte VectorXd -> buffer contíguo (malloc). Use std::free depois.
qpOASES::real_t* to_buffer(const Eigen::VectorXd& v);
