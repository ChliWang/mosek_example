
#include <iostream>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

int main(int argc, char *argv[])
{
    //todo https://docs.mosek.com/latest/cxxfusion/tutorial-cqo-shared.html
    //* conic optimization problem
    Model::t M = new Model("cqo1");  auto _M = finally([&]() { M->dispose(); }); // 出了作用域后自动销毁

    Variable::t x = M->variable("x", 3, Domain::greaterThan(0.0));
    Variable::t y = M->variable("y", 3, Domain::unbounded());

    // create the aliases z1 z2
    // Create the aliases
    //      z1 = [ y[0],x[0],x[1] ]
    //  and z2 = [ y[1],y[2],x[2] ]
    Variable::t z1 = Var::vstack(y->index(0), x->slice(0, 2));
    Variable::t z2 = Var::vstack(y->slice(1, 3), x->slice(0, 2));
    
    // Create the constraint
    //      x[0] + x[1] + 2.0 x[2] = 1.0
    auto aval = new_array_ptr<double, 1>({1.0, 1.0, 2.0});
    M->constraint("lc", Expr::dot(aval, x), Domain::equalsTo(1.0));

    // Create the constraints
    //      z1 belongs to C_3
    //      z2 belongs to K_3
    // where C_3 and K_3 are respectively the quadratic and
    // rotated quadratic cone of size 3, i.e.
    //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    //  and  2.0 z2[0] z2[1] >= z2[2]^2
    Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());
    Constraint::t qc2 = M->constraint("qc2", z2, Domain::inRotatedQCone());

    // Set the objective function to (y[0] + y[1] + y[2])
    M->objective("obj", ObjectiveSense::Minimize, Expr::sum(y));

    M->solve();

    ndarray<double, 1> xlvl  = *(x->level());
    ndarray<double, 1> ylvl  = *(y->level());
    
    ndarray<double, 1> qc1lvl  = *(qc1->level());
    ndarray<double, 1> qc1dual  = *(qc1->dual());

    ndarray<double, 1> qc2lvl  = *(qc1->level());
    ndarray<double, 1> qc2dl  = *(qc1->dual());

    std::cout << "x1,x2,x2 = " << xlvl << std::endl;
    std::cout << "y1,y2,y3 = " << ylvl << std::endl;
    std::cout << std::endl;
    
    std::cout << "qc1 levels = " << qc1lvl << std::endl;
    std::cout << "qc1 dual conic var levels = " << qc1dual << std::endl;
    std::cout << std::endl;

    std::cout << "qc2 levels = " << qc2lvl << std::endl;
    std::cout << "qc2 dual conic var levels = " << qc2dl << std::endl;

    return 0;
}
