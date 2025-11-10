#pragma once
// Interface mínima do controlador usada pelo Communicator

#include <eigen3/Eigen/Dense>

struct IController {
    virtual ~IController() = default;
    virtual Eigen::VectorXd computeU(const Eigen::Ref<const Eigen::VectorXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& xref) = 0;
};
