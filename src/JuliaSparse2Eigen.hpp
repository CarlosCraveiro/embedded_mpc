#pragma once
#include <eigen3/Eigen/Sparse>
#include <vector>

class JuliaSparse2Eigen {
public:
  static Eigen::SparseMatrix<double>
  convert(const std::vector<int>& I1,   // 1-based rows (Julia)
          const std::vector<int>& J1,   // 1-based cols (Julia)
          const std::vector<double>& V, // values
          int m, int n)                 // size
  {
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(V.size());
    for (size_t k = 0; k < V.size(); ++k) {
      T.emplace_back(I1[k] - 1, J1[k] - 1, V[k]); // 0-based for Eigen
    }
    Eigen::SparseMatrix<double> A(m, n);
    A.setFromTriplets(T.begin(), T.end()); // sums duplicates like Julia
    return A;
  }
};

