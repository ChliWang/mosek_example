#include <iostream>

#include "sdqp/sdqp.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    int m = 5;
    Eigen::Matrix<double, 3, 3> Q, Q_;
    Eigen::Matrix<double, 3, 1> c, c_;
    Eigen::Matrix<double, 3, 1> x, temp_x;        // decision variables
    temp_x.setZero();
    Eigen::Matrix<double, -1, 3> A(m, 3); // constraint matrix
    Eigen::VectorXd b(m);                 // constraint bound

    Eigen::Matrix<double, 3, 3> unitMatrix33;
    unitMatrix33.setIdentity();
    // std::cout << unitMatrix33.transpose() << std::endl;
    Q << 8.0, -6.0, 2.0, -6.0, 6.0, -3.0, 2.0, -3.0, 2.0;
    c << 1.0, 3.0, -2.0;

    A << 0.0, -1.0, -2.0,
        -1.0, 1.0, -3.0,
        1.0, -2.0, 0.0,
        -1.0, -2.0, -1.0,
        3.0, 5.0, 1.0 ;
    b << -1.0, 2.0, 7.0, 2.0, -1.0;
    int step = 100000;
    double rho = 1;
    double beta = 1000;
    double gamma = 1;
    double minobj;
    while (--step >= 0)
    {
        Q_ = Q + unitMatrix33 / rho;
        c_ = c + temp_x / rho;
        minobj = sdqp::sdqp<3>(Q_, c_, A, b, x);
        temp_x = x;
        rho = min((1 + gamma)*rho, beta); 
    }
    minobj -= 0.5 / rho * x.squaredNorm();

    std::cout << "optimal sol: " << x.transpose() << std::endl;
    std::cout << "optimal obj: " << minobj << std::endl;
    std::cout << "cons precision: " << (A * x - b).maxCoeff() << std::endl;

    return 0;
}
