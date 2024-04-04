#include <iostream>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;
//todo https://docs.mosek.com/latest/cxxfusion/tutorial-lo-shared.html#doc-tutorial-lo

int main(int argc, char *argv[])
{
    //* Linear Optimization
    Model::t M = new Model("lo1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

    auto A1 = new_array_ptr<double, 1>({3.0, 1.0, 2.0, 0.0});
    auto A2 = new_array_ptr<double, 1>({2.0, 1.0, 3.0, 1.0});
    auto A3 = new_array_ptr<double, 1>({0.0, 2.0, 0.0, 3.0});

    auto c = new_array_ptr<double, 1>({3.0, 1.0, 5.0, 1.0});

    // 
    Variable::t x = M->variable("x", 4, Domain::greaterThan(0.0));
    M->constraint(x->index(1), Domain::lessThan(10.0));

    M->constraint("c1", Expr::dot(A1, x), Domain::equalsTo(30.0));
    M->constraint("c2", Expr::dot(A2, x), Domain::greaterThan(15.0));
    M->constraint("c3", Expr::dot(A3, x), Domain::lessThan(25.0));

    M->objective("obj", ObjectiveSense::Maximize, Expr::dot(c, x));

    M->solve();

    auto sol = x->level();
    std::cout << "[x0,x1,x2,x3] = " << *sol << "\n";

    return 0;
}
