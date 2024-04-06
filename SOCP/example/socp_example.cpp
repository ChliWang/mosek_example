#include <iostream>
#include "lbfgs_raw.hpp"
#include <Eigen/Eigen>
#include <iomanip>
#include <ctime>
using namespace std;
using namespace Eigen;

class socp_example
{
private:
    /* data */
    double PM_beta_ = 1e3;
    double PM_rho_ = 1;
    double PM_gamma_ = 1;
    const double d_ = 1;
    Eigen::VectorXd b_, c, b, f;
    Eigen::VectorXd PM_mu_;
    Eigen::MatrixXd A, A_, H, I_m;
    double minObjective = Infinity;
    // 
    static constexpr int m = 7; // var num []
    static constexpr int N_ = m + 1;
    
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
    
    int run()
    {
        init_param(); //  static constexpr int m = 7, N = m+1;
        double* x = new double[m];
        Eigen::Map<Eigen::VectorXd> inputs(x, m); // x为变量，m = 2  N_= 40 horzion 
        inputs.setZero();
        // 
        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        lbfgs_params.mem_size = 16;
        lbfgs_params.past = 0;
        lbfgs_params.g_epsilon = 0.1; // 1e-4 req_e_prec
        lbfgs_params.min_step = 1e-32;
        lbfgs_params.delta = 1e-5; 
        lbfgs_params.line_search_type = 0;
        
        PM_mu_.resize(N_); // 约束维度
        PM_mu_.setZero();
        PM_rho_ = 1;
        PM_gamma_ = 1;
        PM_beta_ = 1e3;
        
        double req_e_cons = 1e-5, req_e_prec = 1e-5;
        double e_cons = 1, e_prec = 1; // e_cons KKT_condition => 对偶变量梯度， e_prec LBFGS无约束优化的精度
        int ret = 0, times = 0;
        bool firstLoop = true;
        std::clock_t start_time = std::clock();
        while(e_cons > req_e_cons || e_prec > req_e_prec || firstLoop)
        {
            if (firstLoop) firstLoop = false;
            ++times;
            ret = lbfgs::lbfgs_optimize(m, x, &minObjective, &objectiveFunc, nullptr, monitorProgress, this, &lbfgs_params);      
            
            Eigen::Map<const Eigen::VectorXd> x_star(x, m);
            Eigen::VectorXd neg_v(N_);

            neg_v = PM_mu_ / PM_rho_ - A_*x_star - b_;
            Eigen::VectorXd proj_neg_v = SOCP_spectral_decomposition(neg_v);

            e_cons = (PM_mu_/PM_rho_ - proj_neg_v).lpNorm<Eigen::Infinity>();
            Eigen::VectorXd grad_x_star = PM_mu_ / PM_rho_ - proj_neg_v;
            e_prec = grad_x_star.lpNorm<Eigen::Infinity>();
            // std::cout << "======================" << e_cons << std::endl;
            // std::cout << "e_cons: " << e_cons << std::endl;
            // std::cout << "e_prec: " << e_prec << std::endl;
            PM_mu_ = SOCP_spectral_decomposition(neg_v * PM_rho_);
            PM_rho_ = std::min(PM_beta_, (1 + PM_gamma_) * PM_rho_);
            lbfgs_params.g_epsilon = std::max(lbfgs_params.g_epsilon * 0.1, 1e-5); // 逐渐提高内层循环prec
        }
        std::clock_t end_time = std::clock();
        double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

        Eigen::Map<const Eigen::VectorXd> opt_x(x, m);
        std::ios old_settings(nullptr);
        old_settings.copyfmt(std::cout);
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "\033[32m" << "ret: " << ret  << std::endl
                  << "mu_sqr_norm/2rho: " << PM_mu_.squaredNorm()/(2*PM_rho_) << std::endl
                  << "optimal sol: " << opt_x.transpose() << std::endl
                  << "optimal obj: " << opt_x.dot(f) << "\033[0m" << std::endl
                  << "optimal time : " << elapsed_time << " second\n";
        std::cout.copyfmt(old_settings);
        std::cout << "e_cons: " << e_cons << std::endl
                  << "e_prec: " << e_prec << std::endl;
        return ret;
    }
    static inline double objectiveFunc(void* ptrObj,
                                     const double* x,
                                     double* grad,
                                     const int n)
    {
        socp_example& obj = *(socp_example*)ptrObj;
        Eigen::Map<const Eigen::VectorXd> inputs(x, m);
        Eigen::Map<Eigen::VectorXd> grad_inputs(grad, m); // 
        
        // cost 
        double totalcost = 0;
        Eigen::VectorXd neg_v(N_); // -v := mu/rho -(Ax + b) 
        neg_v = obj.PM_mu_ / obj.PM_rho_ - obj.A_*inputs - obj.b_;
        Eigen::VectorXd proj_neg_v = obj.SOCP_spectral_decomposition(neg_v);
        totalcost += inputs.dot(obj.f);
        totalcost += obj.PM_rho_ / 2 * proj_neg_v.squaredNorm();
        
        // grad 
        grad_inputs.setZero();
        grad_inputs += obj.f;
        // obj.B_subdifferential(temp * obj.PM_rho_); // get Hessian H
        // grad_inputs += obj.PM_rho_ * obj.A_.transpose() * obj.H * obj.A_;
        // grad_inputs += obj.A_.transpose() * obj.PM_rho_* proj_neg_v;
        Eigen::VectorXd proj_rho_neg_v = obj.SOCP_spectral_decomposition(obj.PM_rho_ * neg_v);
        grad_inputs += -obj.A_.transpose() * proj_rho_neg_v; // m * 1
        return totalcost;
    }
    inline void B_subdifferential(Eigen::VectorXd temp_B) // LP
    {
        double x1 = temp_B(0);
        Eigen::VectorXd temp_x2 = temp_B.segment(1, m);
        double x2_norm = temp_x2.norm();
        if (x2_norm <= x1) H.setIdentity();
        else if(x2_norm <= -x1) H.setZero();
        else if (x2_norm > abs(x1)) 
        {
            H(0, 0) = 0.5;
            H.block(0, 1, 1, m) = temp_x2 / (2 * x2_norm);
            H.block(1, 0, m, 1) = temp_x2 / (2 * x2_norm);
            H.block(1, 1, m, m) = (x1 + x2_norm)/(2*x2_norm) * I_m - (x1 * pow(x2_norm,2))/pow(x2_norm,3) * I_m; 
        }   
    }
    inline Eigen::VectorXd SOCP_spectral_decomposition(Eigen::VectorXd v) // LP
    {
        int v_size = v.size(); // N_
        Eigen::VectorXd P_k(v_size);
        double v0, v1_norm;
        v0 = v(0);
        Eigen::VectorXd temp_v = v.segment(1, m);
        v1_norm = temp_v.norm();
        if (v0 <= -v1_norm) P_k.setZero();
        else if(v0 >= v1_norm) P_k = v;
        else  // (abs(v0) < v1_norm)
        {
            double coeff = (v0 + v1_norm)/(2*v1_norm);
            P_k = v;
            P_k(0) = v1_norm;
            P_k *= coeff;
        }
        return P_k;
    }
    
    inline void spectral_decomposition(Eigen::VectorXd& v) // LP
    {
        // int n_v = v.size();
        // 创建对角矩阵，以特征值作为主对角线
        Eigen::MatrixXd D = v.asDiagonal();
        // 计算特征分解
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(D);
        if (eigensolver.info() != Eigen::Success)
        {
            std::cerr << "Eigen decomposition failed." << std::endl;
            return;
        }
        // 获取特征值和特征向量
        Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
        std::cout << "Eigenvalues:" << std::endl << eigenvalues << std::endl;
        std::cout << "Eigenvectors:" << std::endl << eigenvectors << std::endl;
        
        // 修正特征值为非负值
        eigenvalues = eigenvalues.cwiseMax(0.0);
        v = eigenvectors * eigenvalues;
    }
    // typedef int (*lbfgs_progress_t)(void *instance,
    //                             const double *x,
    //                             const double *g,
    //                             const double fx,
    //                             const double xnorm,
    //                             const double gnorm,
    //                             const double step,
    //                             int n,
    //                             int k,
    //                             int ls);
    static int monitorProgress(void *instance,
                                const double *x,
                                const double *g,
                                const double fx,
                                const double xnorm,
                                const double gnorm,
                                const double step,
                                int n,
                                int k,
                                int ls)
    {
        // Eigen::Map<Eigen::VectorXd> grad_(g, n);
        // std::cout << std::setprecision(4)
                //   << "================================" << std::endl
        //           << "Iteration: " << k << std::endl
                //   << "Function Value: " << fx << std::endl;
                //   << "Gradient Inf Norm: " << grad_.cwiseAbs().maxCoeff() << std::endl;
                //   << "Variables: " << std::endl
                //   << x.transpose() << std::endl;
        return 0;
    }
};

int main(int argc, char **argv)
{
    socp_example socp_test;
    return socp_test.run();
}