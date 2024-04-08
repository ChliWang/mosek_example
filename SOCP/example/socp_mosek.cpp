#include <iostream>
#include <Eigen/Eigen>
#include <iomanip>
#include <iterator>

#include <fusion.h>
#include <exception>

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

        //* From Eigen::MatrixXd 间接定义A_def1
        std::shared_ptr<ndarray<double,2>> A_def1 = std::make_shared<ndarray<double,2>>(shape(A.rows(), A.cols()));
        for (int i = 0; i < A.rows(); i++) 
            for (int j = 0; j < A.cols(); j++) 
                (*A_def1)(i, j) = A(i, j);
        std::cout << "A_def1 " << std::endl;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) 
                std::cout << (*A_def1)(i, j) << " ";
            std::cout << std::endl;
        }        
        //* 间接定义A_def2
        std::shared_ptr<ndarray<double,1>> A_def2 = std::make_shared<ndarray<double,1>>(shape(A.rows() * A.cols()));
        for (int i = 0; i < A.rows(); i++) 
            for (int j = 0; j < A.cols(); j++) 
                (*A_def2)(i * m + j) = A(i, j);
        std::cout << "A_def2 " << std::endl;
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) 
                std::cout << (*A_def2)(i*m+j) << " ";
            std::cout << std::endl;
        }  
        //* 直接定义A_def
        auto A_def  = new_array_ptr<double, 1>
                        ({7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
        //* 直接定义b_m, f_m
        auto b_m = new_array_ptr<double, 1>({1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0});
        auto f_m = new_array_ptr<double, 1>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
        std::cout << "b_m " << std::endl;
        printsol(b_m);
        std::cout << "f_m " << std::endl;
        printsol(f_m);
        //* From std::vector<double> 间接定义b_m2, f_m2
        auto b_m2 = dblarray({1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0}); // Direct initialization
        auto f_m2 = ndou(std::vector<double>(c.data(), c.data() + c.size())); // Convert Eigen::VectorXd to std::vector
        std::cout << "b_m2 " << std::endl;
        printsol(b_m2);
        std::cout << "f_m2 " << std::endl;
        printsol(f_m2);
        // printsol(f_m);
        M->constraint(Expr::vstack(right, Expr::add(Expr::mul(Matrix::dense(7, 7, A_def2), x), b_m2)), Domain::inQCone());
        M->objective(ObjectiveSense::Minimize, Expr::dot(f_m2, x) );
        // setupExample(M);
        try
        {
            // M->setLogHandler([ = ](const std::string & msg) { std::cout << msg << std::flush; } ); 
            M->setSolverParam("mioTolRelGap", 1e-5); 
            M->solve();

            // We expect solution status OPTIMAL (this is also default)
            M->acceptedSolutionStatus(AccSolutionStatus::Optimal);
            printsol(x->level());
            double tm = M->getSolverDoubleInfo("optimizerTime");
            int it = M->getSolverIntInfo("intpntIter");
            std::cout << "Time: " << tm << "\nIterations: " << it << "\n";
        }
        catch (const OptimizeError& e)
        {
            std::cout << "Optimization failed. Error: " << e.what() << "\n";
        }
        catch (const SolutionError& e)
        {
            // The solution with at least the expected status was not available.
            // We try to diagnoze why.
            std::cout << "Requested solution was not available.\n";
            auto prosta = M->getProblemStatus();
            switch(prosta)
            {
            case ProblemStatus::DualInfeasible:
                std::cout << "Dual infeasibility certificate found.\n";
                break;

            case ProblemStatus::PrimalInfeasible:
                std::cout << "Primal infeasibility certificate found.\n";
                break;

            case ProblemStatus::Unknown:
                // The solutions status is unknown. The termination code
                // indicates why the optimizer terminated prematurely.
                std::cout << "The solution status is unknown.\n";
                char symname[MSK_MAX_STR_LEN];
                char desc[MSK_MAX_STR_LEN];
                // MSK_getcodedesc((MSKrescodee)(M->getSolverIntInfo("optimizeResponse")), symname, desc);//? 有问题...
                // std::cout << "  Termination code: " << symname << " " << desc << "\n";
                break;

            default:
                std::cout << "Another unexpected problem status: " << prosta << "\n";
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Unexpected error: " << e.what() << "\n";
        }

        M->dispose();
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
