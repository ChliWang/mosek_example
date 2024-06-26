//
//    Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//    File:    mioinitsol.cc
//
//    Purpose:  Demonstrates how to solve a small mixed
//              integer linear optimization problem
//              providing an initial feasible solution.
//
#include <memory>
#include <iostream>


#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

int main(int argc, char ** argv)
{

  Model::t M = new Model("lo1"); auto _M = finally([&]() { M->dispose(); });
  M->setLogHandler([ = ](const std::string & msg) { std::cout << msg << std::flush; } ); // 设置log

  auto c  = new_array_ptr<double, 1>({7.0, 10.0, 1.0, 5.0});
  int n = c->size();

  auto x = M->variable("x", n, Domain::greaterThan(0.0));
  x->slice(0,3)->makeInteger(); // x0 x1 x2 是整数类型

  M->constraint(Expr::sum(x), Domain::lessThan(2.5));

  M->objective("obj", ObjectiveSense::Maximize, Expr::dot(c, x));

  // Assign values to integer variables.
  // We only set a slice of x     
  auto init_sol = new_array_ptr<double, 1>({ 1.0, 1.0, 0.0 });
  x->slice(0,3)->setLevel( init_sol );

  // Request constructing the solution from integer variable values
  M->setSolverParam("mioConstructSol", "on");

  M->solve();

  auto ss = M->getPrimalSolutionStatus();
  std::cout << "Solution status: " << ss << std::endl;
  auto sol = x->level();
  std::cout << "x = ";

  for (auto s : *sol)
    std::cout << s << ", ";

  // Was the initial solution used?
  int constr = M->getSolverIntInfo("mioConstructSolution");
  double constrVal = M->getSolverDoubleInfo("mioConstructSolutionObj");
  std::cout << "Construct solution utilization: " << constr << std::endl;
  std::cout << "Construct solution objective: " << constrVal << std::endl;

}
