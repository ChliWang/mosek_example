
#include <iostream>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

int main(int argc, char *argv[])
{
    //todo https://docs.mosek.com/latest/cxxfusion/tutorial-pow-shared.html

    Model::t M = new Model("pow1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

    Variable::t x = M->variable("x", 3, Domain::unbounded());
    Variable::t x3 = M->variable();
    Variable::t x4 = M->variable();

    // Create the linear constraint
    auto aval = new_array_ptr<double, 1>({1.0, 1.0, 0.5});
    M->constraint(Expr::dot(x, aval), Domain::equalsTo(2.0));

    // Create the conic constraints
    M->constraint(Var::vstack(x->slice(0, 2), x3), Domain::inPPowerCone(0.2)); // 堆叠变量
    M->constraint(Expr::vstack(x->index(2), 1.0, x4), Domain::inPPowerCone(0.4)); // 堆叠表达式不限于变量
    
    

    return 0;
}
