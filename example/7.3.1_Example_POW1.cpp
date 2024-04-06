
#include <iostream>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

int main(int argc, char *argv[])
{
    //todo https://docs.mosek.com/latest/cxxfusion/tutorial-ceo-shared.html
    //* Power Cone Optimization
    
    Model::t M = new Model("pow1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

    Variable::t x = M->variable("x", 3, Domain::unbounded());

    // Create the constraint
    //      x[0] + x[1] + x[2] = 1.0
    M->constraint("lc", Expr::sum(x), Domain::equalsTo(1.0));

    // Create the exponential conic constraint
    Constraint::t expc = M->constraint("expc", x, Domain::inPExpCone());

    M->objective("obj", ObjectiveSense::Minimize, Expr::sum(x->slice(0,2)));

    M->solve();

    // Get the linear solution values
    ndarray<double, 1> xlevel = *(x->level());

    // Get conic solution of expc1
    ndarray<double, 1> expclevel = *(expc->level());
    ndarray<double, 1> expcdual = *(expc->dual());

    std::cout << "x1,x2,x3 = " << xlevel << std::endl;
    std::cout << "expc levels = " << expclevel << std::endl; // 指数锥约束对应的指数变量的取值情况 当前为x1,x2,x3
    std::cout << "expc dual conic var levels = " << expcdual << std::endl; // 

    return 0;
}
