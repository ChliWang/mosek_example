#include <iostream>
#include <Eigen/Eigen>
#include <iomanip>
#include <iterator>

#include <fusion.h>

using namespace mosek::fusion;
using namespace monty;

using namespace std;

class socp_example
{
private:
    /* data */

    double d_ = 1;
    Eigen::VectorXd b_, c, b, f;
    Eigen::VectorXd PM_mu_;
    Eigen::MatrixXd A, A_, H, I_m;
    
    // 
    static constexpr int m = 7; // var 
    static constexpr int N_ = m + 1; // constraint
    
    std::shared_ptr<ndarray<double,2>> A_m = std::make_shared<ndarray<double,2>>(shape(m, m));
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
    
    std::shared_ptr<ndarray<double,1>> ndou(const std::vector<double> &X) {
        return new_array_ptr<double>(X);
    }
    
    std::shared_ptr<ndarray<double,1>> dblarray(std::initializer_list<double> x) {
        return new_array_ptr<double,1>(x); 
    }
    
    void printsol(const std::shared_ptr<ndarray<double, 1>> & a) 
    {
        std::cout << "x = ";
        for(auto val : *a) std::cout << val << " ";
        std::cout << "\n";
    }
    
    void run_mosek()
    {
        Model::t M = new Model("socp");  auto _M = finally([&]() { M->dispose(); }); 
        int m = 7;
        // // Initialize A_m
        // for(int i = 0; i < m; i ++)
        //     (*A_m)(i, i)  = A(i, i); // Assign values 
        
        // cout << "(*A_m)(1, 1) " << (*A_m)(1, 1) << endl;
        Variable::t x  =  M->variable(m, Domain::unbounded());
        Expression::t right = Expr::add(x->index(0), 1.0);

        // auto A_def  = Matrix::dense(new_array_ptr<double, 2>
                        // ({{7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
                        //   {0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
                        //   {0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0}, 
                        //   {0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0}, 
                        //   {0.0, 6.0, 0.0, 0.0, 3.0, 0.0, 0.0}, 
                        //   {0.0, 6.0, 0.0, 0.0, 0.0, 2.0, 0.0}, 
                        //   {0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 1.0}})); // new_array_ptr<double, rank> rank = 2表示维数有2维

        auto A_def  = new_array_ptr<double, 1>
                        ({7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        
        // auto b = dblarray({1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0}); // Direct initialization
        // auto f_m = ndou(std::vector<double>(c.data(), c.data() + c.size())); // Convert Eigen::VectorXd to std::vector
        // printsol(f_m);

        auto b_m = new_array_ptr<double, 1>({1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0});
        auto f_m = new_array_ptr<double, 1>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
        printsol(b_m);
        printsol(f_m);
        M->constraint(Expr::vstack(right, Expr::add(Expr::mul(Matrix::dense(m, m, A_def), x), b_m)), Domain::inQCone());
        M->objective(ObjectiveSense::Minimize, Expr::dot(f_m, x) );
        // try
        // {
        M->setLogHandler([ = ](const std::string & msg) { std::cout << msg << std::flush; } ); 

        M->setSolverParam("mioTolRelGap", 1e-5); 
        M->solve();
        printsol(x->level());
        double tm = M->getSolverDoubleInfo("optimizerTime");
        int it = M->getSolverIntInfo("intpntIter");
        std::cout << "Time: " << tm << "\nIterations: " << it << "\n";
        // }
        // catch (...) {}
    }
    
    void run()
    {
        init_param(); 
        std::cout << " init_param " << std::endl;
        run_mosek();
    }
};

int main(int argc, char **argv)
{
    socp_example socp_test;
    socp_test.run();
    return 0;
}
