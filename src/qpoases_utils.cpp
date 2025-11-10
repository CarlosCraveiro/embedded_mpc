#include "qpoases_utils.hpp"

#include <iostream>   // std::cout
#include <cstdlib>    // std::malloc, std::free
#include <algorithm>  // std::fill_n
#include <new>        // std::bad_alloc

void print_matriz_qpoases(const qpOASES::real_t* A,
                          std::size_t rows,
                          std::size_t cols)
{
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            std::cout << A[r * cols + c];
            if (c + 1 < cols) std::cout << ' ';
        }
        std::cout << '\n';
    }
}

qpOASES::real_t* to_rowwise_dense(const Eigen::SparseMatrix<double>& B) {
    const int m = B.rows();
    const int n = B.cols();

    const std::size_t len = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    auto* buf = static_cast<qpOASES::real_t*>(std::malloc(len * sizeof(qpOASES::real_t)));
    if (!buf) throw std::bad_alloc{};

    std::fill_n(buf, len, static_cast<qpOASES::real_t>(0));

    // B.makeCompressed(); // opcional se já estiver comprimida

    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            const int i = it.row();
            const int j = it.col();
            buf[static_cast<std::size_t>(i) * static_cast<std::size_t>(n)
              + static_cast<std::size_t>(j)] = static_cast<qpOASES::real_t>(it.value());
        }
    }
    return buf; // std::free(buf) depois
}

qpOASES::real_t* to_buffer(const Eigen::VectorXd& v) {
    const Eigen::Index n = v.size();
    auto* buf = static_cast<qpOASES::real_t*>(
        std::malloc(static_cast<std::size_t>(n) * sizeof(qpOASES::real_t)));
    if (!buf) throw std::bad_alloc{};

    for (Eigen::Index i = 0; i < n; ++i) {
        buf[i] = static_cast<qpOASES::real_t>(v[i]);
    }
    return buf; // std::free(buf) depois
}
