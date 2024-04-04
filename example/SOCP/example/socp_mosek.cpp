#include <iostream>
#include "lbfgs_raw.hpp"
#include <Eigen/Eigen>
#include <iomanip>

#include <fusion.h>
using namespace mosek::fusion;
using namespace monty;

using namespace std;
using namespace Eigen;

class socp_example
{
private:
    /* data */
    double PM_beta_ = 1e3;
    double PM_rho_ = 1;
    double PM_gamma_ = 1;
    double d_ = 1;
    Eigen::VectorXd b_, c, b, f;
    Eigen::VectorXd PM_mu_;
    Eigen::MatrixXd A, A_, H, I_m;
    double minObjective = Infinity;
    // 
    static constexpr int m = 7; // var 
    static constexpr int N_ = m + 1; // constraint
    
public:
    void init_param()
    {
        A.resize(m, m);
        A_.resize(N_, m);
        b.resize(m);
        b_.resize(N_);
        b_.setZero();
        c.resize(m);
        f.resize(m);
        A << 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
        b << 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0;

        c << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

        f << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;

        A_.block(0, 0, 1, m) = c.transpose();
        A_.block(1, 0, m, m) = A;
        b_(0) = d_;
        b_.segment(1, m) = b;

        I_m.resize(m, m);
        I_m.setIdentity();

        H.resize(N_, N_); // 
        return;
    }
    Eigen::VectorXd run_mosek()
    {
        Model::t M = new Model("cqo1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

        Variable::t x = M->variable("x", m, Domain::greaterThan(0.0));

        //? Eigen 如何 与 mosk 转化？
        // Expr::t Ax_b = Expr::add(Expr::mul(A, x), b);

        // Expr::t cx_d = Expr::dot(c, x) + d_;

        // Variable::t z1 = Var::vstack(cx_d, Ax_b);

        // Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());

        // Matrix::t fx = f.dot(x);

        // M->objective("obj", ObjectiveSense::Minimize, fx);

        M->solve();
        Eigen::VectorXd res;
        // ndarray<double, 1> xlevel  = *(x->level());
        
        return res;
    }
    
    void run()
    {
        init_param(); //  static constexpr int m = 7, N = m+1;
        
        Eigen::VectorXd opt_x = run_mosek();
    
        std::cout << opt_x << std::endl;
    }

    Eigen::VectorXd ndarrayToEigen(const ndarray<double, 1>& array) 
    {
        Eigen::VectorXd eigenVec(array.size());
        for (size_t i = 0; i < array.size(); ++i) {
            eigenVec(i) = array[i];
        }
        return eigenVec;
    }
};

int main(int argc, char **argv)
{
    socp_example socp_test;
    return socp_test.run();
}