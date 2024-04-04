
#include <iostream>
#include "fusion.h"
#include <iomanip>

using namespace mosek::fusion;
using namespace monty;

// Models log(sum(exp(Ax+b))) <= 0.
// Each row of [A b] describes one of the exp-terms
void logsumexp(Model::t                             M, 
               std::shared_ptr<ndarray<double, 2>>  A, 
               Variable::t                          x,
               std::shared_ptr<ndarray<double, 1>>  b)
{
  int k = A->size(0);
  auto u = M->variable(k); // 
  M->constraint(Expr::sum(u), Domain::equalsTo(1.0)); // sum(u_i) = 1 多个constraint不加具体名称
  M->constraint(Expr::hstack(u,
                             Expr::constTerm(k, 1.0),
                             Expr::add(Expr::mul(A, x), b)), Domain::inPExpCone()); // 2个affine conic Exp cons
}

std::shared_ptr<ndarray<double, 1>> max_volume_box(double Aw, double Af, 
                                                   double alpha, double beta, double gamma, double delta)
{
  Model::t M = new Model("max_vol_box"); auto _M = finally([&]() { M->dispose(); });

  auto xyz = M->variable(3);
  M->objective("Objective", ObjectiveSense::Maximize, Expr::sum(xyz));
    
  logsumexp(M, 
            new_array_ptr<double,2>({{1,1,0}, {1,0,1}}), 
            xyz, 
            new_array_ptr<double,1>({log(2.0/Aw), log(2.0/Aw)}));
    
  M->constraint(Expr::dot(new_array_ptr<double,1>({0,1,1}), xyz), Domain::lessThan(log(Af)));
  M->constraint(Expr::dot(new_array_ptr<double,1>({1,-1,0}), xyz), Domain::inRange(log(alpha),log(beta)));
  M->constraint(Expr::dot(new_array_ptr<double,1>({0,-1,1}), xyz), Domain::inRange(log(gamma),log(delta)));
    
  M->setLogHandler([](const std::string & msg) { std::cout << msg << std::flush; } );
  M->solve();
  // 捕获外部变量xyz, 循环遍历xyz, *(xyz->level()) = x y z的值，通过return exp(x / y / z)恢复 h w d
  return std::make_shared<ndarray<double, 1>>(shape(3), [xyz](ptrdiff_t i) { return exp((*(xyz->level()))[i]); });
}

int main(int argc, char *argv[])
{
    //todo https://docs.mosek.com/latest/cxxfusion/tutorial-gp-shared.html
    //* Geometric Programming
    //* 考虑最大化盒子的体积 hwd 

    double Aw    = 200.0;
    double Af    = 50.0;
    double alpha = 2.0;
    double beta  = 10.0;
    double gamma = 2.0;
    double delta = 10.0;
    // std::shared_ptr<monty::ndarray<double, 1>> hwd
    auto hwd = max_volume_box(Aw, Af, alpha, beta, gamma, delta); // 6 个 Scalar variables  

    std::cout << std::setprecision(4);
    std::cout << "h=" << (*hwd)[0] << " w=" << (*hwd)[1] << " d=" << (*hwd)[2] << std::endl;
    return 0;
}

//! 注意单项式变量替换的方法：该例子中表现出是分数相乘，转换为ExpCone