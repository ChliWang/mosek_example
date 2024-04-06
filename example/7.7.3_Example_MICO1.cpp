#include <iostream>
#include <iomanip>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

int main(int argc, char *argv[])
{
    //! 看见e就要想起来 conic Exp Cone
    Model::t M = new Model("mico1"); auto _M = finally([&]() { M->dispose(); });

    Variable::t x = M->variable(Domain::integral(Domain::unbounded()));
    Variable::t y = M->variable(Domain::integral(Domain::unbounded()));
    Variable::t t = M->variable();
    
    M->constraint(Expr::vstack(t, x, y), Domain::inQCone());
    M->constraint(Expr::vstack(Expr::sub(x, 3.8), 1, y), Domain::inPExpCone());

    M->objective(ObjectiveSense::Minimize, t);

    M->solve();

    std::cout << std::setprecision(2)
                << "x = " << (*(x->level()))[0] << std::endl
                << "y = " << (*(y->level()))[0] << std::endl ;

    return 0;
}
