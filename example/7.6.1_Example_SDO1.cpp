#include <iostream>
#include "fusion.h"
#include <Eigen/Eigen>

using namespace mosek::fusion;
using namespace monty;
using namespace std;

std::shared_ptr<ndarray<double,1>> ndou(const std::vector<double> &X) { return new_array_ptr<double>(X); }

int main(int argc, char *argv[])
{
    //* 7.6 Semidefinite Optimization
    Model::t M = new Model("sdp1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

    // Setting up the variables
    Variable::t X  = M->variable("X", Domain::inPSDCone(3));
    Variable::t x  = M->variable("x", Domain::inQCone(3));

    // Setting up the constant coefficient matrices
    // Eigen::MatrixXd tt(3, 3);
    // tt << 2.0, 1.0, 0.0,
    //       1.0, 2.0, 1.0,
    //       0.0, 1.0, 2.0;
    // std::vector<vector<double>> tt1;
    // for (int i = 0; i < tt.rows(); ++i) 
    // {
    //     std::vector<double> row;
    //     for (int j = 0; j < tt.cols(); ++j) 
    //     {
    //         row.push_back(tt(i, j));
    //     }
    //     tt1.push_back(row);
    // }
    // std::cout << tt1[1][1] << endl;
    // auto tt1_ = ndou(tt1[0]);
    // std::cout << *tt1_ << endl;

    Matrix::t C  = Matrix::dense(new_array_ptr<double, 2>({{2., 1., 0.}, {1., 2., 1.}, {0., 1., 2.}}));
    Matrix::t A1 = Matrix::eye(3);
    Matrix::t A2 = Matrix::ones(3, 3);
    // double A2_00 = A2->get(0, 0);
    // std::cout << A2_00 << std::endl;
    // Objective
    M->objective(ObjectiveSense::Minimize, Expr::add(Expr::dot(C, X), x->index(0))); // 矩阵点积 对应元素相乘 并累加

    // Constraints
    M->constraint("c1", Expr::add(Expr::dot(A1, X), x->index(0)), Domain::equalsTo(1.0));
    M->constraint("c2", Expr::add(Expr::dot(A2, X), Expr::sum(x->slice(1, 3))), Domain::equalsTo(0.5));

    M->solve();

    std::cout << "Solution : " << std::endl;
    std::cout << "  X = " << *(X->level()) << std::endl;
    std::cout << "  x = " << *(x->level()) << std::endl;

    return 0;
}
