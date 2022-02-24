#ifndef CDL012Exponential_H
#define CDL012Exponential_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"
#include <algorithm>

template <class T>
class CDL012Exponential : public CD<T, CDL012Exponential<T>> {
    private:
        const double LipschitzConst = 0.25;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        arma::vec ExpyXB;
        // std::vector<double> * Xtr;
        arma::uvec indices;

        // T * Xy;

        // arma::vec inverse_ExpyXB;
        // double d_minus;
        // double current_expo_loss;

        //////// new variables for expo loss
        arma::vec inverse_ExpyXB;
        double d_minus;
        double d_plus;
        double current_expo_loss;
        std::unordered_map<std::size_t, arma::uvec> * Xy_neg_indices;
        T * Xy;

        
    public:
        CDL012Exponential(const T& Xi, const arma::vec& yi, const Params<T>& P);
        //~CDL012Logistic(){}
        
        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;
        
        inline double GetBiGrad(const std::size_t i);
        
        inline double GetBiValue(const double old_Bi, const double grd_Bi);
        
        inline double GetBiReg(const double Bi_step);
        
        inline void ApplyNewBi(const std::size_t i, const double Bi_old, const double Bi_new);
        
        inline void ApplyNewBiCWMinCheck(const std::size_t i, const double old_Bi, const double new_Bi);

        // new special functions for exponential loss
        void UpdateBi(const std::size_t i);
        bool CWMinCheck();
        bool UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin);

};

template <class T>
inline double CDL012Exponential<T>::GetBiGrad(const std::size_t i){
    /*
     * Notes:
     *      When called in CWMinCheck, we know that this->B[i] is 0.
     */
    // return -arma::dot(matrix_column_get(*(this->Xy), i), 1 / (1 + ExpyXB) ) //+ twolambda2 * this->B[i];
    // indices = arma::find(matrix_column_get(*(this->Xy), i) < 0);
    indices = (*(this->Xy_neg_indices))[i];
    this->d_minus = arma::sum(this->inverse_ExpyXB.elem(indices))/this->current_expo_loss;
    this->d_minus = this->d_minus <= 1e-10 ? 1e-10 : this->d_minus >= 1-1e-10 ? 1-1e-10 : this->d_minus;
    // if ((this->d_minus < 1e-9) || (this->d_minus > 1 - 1e-9)) {
    //     std::cout << "Finally I find the cause!!!\n";
    //     // this->d_minus = this->d_minus <= 1e-9 ? 1e-9 : this->d_minus >= 1-1e-9 ? 1-1e-9 : this->d_minus;
    // }

    
    // this->d_minus = arma::sum(this->inverse_ExpyXB % matrix_column_get(*(this->Xy), i))/this->current_expo_loss;

    return -0.5*std::log((1.0-this->d_minus)/this->d_minus);
    //return -arma::sum( matrix_column_get(*(this->Xy), i) / (1 + ExpyXB) ) + twolambda2 * this->B[i];
}

template <class T>
inline double CDL012Exponential<T>::GetBiValue(const double old_Bi, const double grd_Bi){
    return old_Bi - grd_Bi/qp2lamda2;
}

template <class T>
inline double CDL012Exponential<T>::GetBiReg(const double Bi_step){
    return std::abs(Bi_step) - lambda1ol;
}

template <class T>
inline void CDL012Exponential<T>::ApplyNewBi(const std::size_t i, const double old_Bi, const double new_Bi){
    this->inverse_ExpyXB %= arma::exp( (old_Bi - new_Bi) * matrix_column_get(*(this->Xy), i));
    this->B[i] = new_Bi;
}

template <class T>
inline void CDL012Exponential<T>::ApplyNewBiCWMinCheck(const std::size_t i,
                                                    const double old_Bi,
                                                    const double new_Bi){
    this->inverse_ExpyXB %= arma::exp( (old_Bi - new_Bi) * matrix_column_get(*(this->Xy), i));
    this->B[i] = new_Bi;
    this->Order.push_back(i);
}

template <class T>
inline double CDL012Exponential<T>::Objective(const arma::vec & inverse_expyXB, const beta_vector & B) {  // hint inline
    // const auto l2norm = arma::norm(B, 2);
    // arma::sum(arma::log(1 + 1 / expyXB)) is the negative log-likelihood
    return arma::sum(inverse_expyXB) + this->lambda0 * n_nonzero(B); //+ this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012Exponential<T>::Objective() {  // hint inline
    // const auto l2norm = arma::norm(this->B, 2);
    // arma::sum(arma::log(1 + 1 / ExpyXB)) is the negative log-likelihood
    return arma::sum(this->inverse_ExpyXB) + this->lambda0 * n_nonzero(this->B); //+ this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
CDL012Exponential<T>::CDL012Exponential(const T& Xi, const arma::vec& yi, const Params<T>& P) : CD<T, CDL012Exponential<T>>(Xi, yi, P) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    lambda1ol = this->lambda1 / qp2lamda2;
    
    this->inverse_ExpyXB = arma::exp(-*this->y % (*(this->X) * this->B + this->b0)); // Maintained throughout the algorithm
    // Xy = P.Xy;
    this->current_expo_loss = arma::sum(this->inverse_ExpyXB);
    Xy_neg_indices = P.Xy_neg_indices;
    Xy = P.Xy;
}

// new function overiding the old function in CD.h
template<class T>
bool CDL012Exponential<T>::UpdateBiCWMinCheck(const std::size_t i, const bool Cwmin){
    const double grd_Bi = GetBiGrad(i);

    double new_loss = this->current_expo_loss*2*std::sqrt(this->d_minus*(1-this->d_minus));
    if (grd_Bi != 0) {
        if (this->current_expo_loss <= new_loss + this->lambda0) {
            return Cwmin;
        } else {
            this->B[i] -= grd_Bi;
            this->inverse_ExpyXB %= arma::exp(matrix_column_get(*(this->Xy), i) * grd_Bi);
            this->current_expo_loss = new_loss;
            this->Order.push_back(i);
            return false;
        }
    } else {
        return Cwmin;
    }
}

// new function overiding the old function in CD.h
template<class T>
bool CDL012Exponential<T>::CWMinCheck() {
    std::vector<std::size_t> S = nnzIndicies(this->B);
    
    std::vector<std::size_t> Sc;
    set_difference(
        this->Range1p.begin(),
        this->Range1p.end(),
        S.begin(),
        S.end(),
        back_inserter(Sc));
    
    bool Cwmin = true;
    for (auto& i : Sc) {
        // Rcpp::Rcout << "CW Iteration: " << i << "\n";
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Cwmin = UpdateBiCWMinCheck(i, Cwmin);
    }
    
    // Rcpp::Rcout << "CWMinCheckL " << Cwmin << "\n";
    
    return Cwmin;
}

template <class T>
void CDL012Exponential<T>::UpdateBi(const std::size_t i) {
    if (this->B[i] != 0) {
        this->inverse_ExpyXB %= arma::exp(matrix_column_get(*(this->Xy), i) * this->B[i]);
        this->current_expo_loss = arma::sum(this->inverse_ExpyXB);
        this->B[i] = 0;
    }
    const double grd_Bi = GetBiGrad(i); // Gradient of Loss wrt to Bi
    double new_loss = this->current_expo_loss*2*std::sqrt(this->d_minus*(1-this->d_minus));
    if (new_loss + this->lambda0 < this->current_expo_loss){
        this->inverse_ExpyXB %= arma::exp(matrix_column_get(*(this->Xy), i) * grd_Bi);
        this->current_expo_loss = new_loss;
        this->B[i] -= grd_Bi;
    }
}

template <class T>
FitResult<T> CDL012Exponential<T>::_Fit() {
    this->objective = Objective(); // Implicitly used ExpyXB
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        // std::cout << "at iteration " + std::to_string(t) + ", the loss is " + std::to_string(Objective()) + "\n";
        this->Bprev = this->B;

        if (this->intercept){
            const double b0old = this->b0;

            // indices = arma::find(*(this->y) < 0);
            indices = (*(this->Xy_neg_indices))[-1];
            // this->d_minus = arma::sum(this->inverse_ExpyXB.elem(indices)) / arma::sum(this->inverse_ExpyXB);
            this->d_minus = arma::sum(this->inverse_ExpyXB.elem(indices)) / this->current_expo_loss;
            const double partial_b0 = -0.5*std::log((1-this->d_minus)/this->d_minus);
            this->b0 -= partial_b0;
            this->inverse_ExpyXB %= arma::exp( partial_b0 * *(this->y));
            this->current_expo_loss *= 2*std::sqrt(this->d_minus*(1-this->d_minus));

            // // const double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
            // const double partial_b0 = - arma::dot( *(this->y) , 1/(1 + ExpyXB) );
            // this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            // ExpyXB %= arma::exp( (this->b0 - b0old) * *(this->y));
        }
        
        for (auto& i : this->Order) {
            // std::cout << "at iteration " + std::to_string(t) + ", after optimizing on coordinate " + std::to_string(i) + ", the loss is " + std::to_string(Objective()) + "\n";
            this->UpdateBi(i);
        }
        // std::cout << "CDL012Logistic.h line 145. Total loss is " + std::to_string(this->current_expo_loss) + " & " + std::to_string(arma::sum(this->inverse_ExpyXB)) + "\n";
        // std::cout << "Another check of total loss is: " + std::to_string(arma::sum(1.0 / arma::exp(*this->y % (*(this->X) * this->B + this->b0)))) + "\n";
        // std::cout << "sparsity level is " + std::to_string(n_nonzero(this->B)) + "\n";

        this->RestrictSupport();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->isConverged() && this->CWMinCheck()) {
            break;
        }
    }
    
    // this->result.Objective = this->objective;
    this->result.Objective = Objective();
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    // this->result.ExpyXB = 1.0 / this->inverse_ExpyXB;
    this->result.inverse_ExpyXB = this->inverse_ExpyXB;
    this->result.IterNum = this->CurrentIters;
    
    return this->result;
}

template <class T>
FitResult<T> CDL012Exponential<T>::_FitWithBounds() { // always uses active sets
    
    //arma::sp_mat B2 = this->B;
    clamp_by_vector(this->B, this->Lows, this->Highs);
    
    this->objective = Objective(); // Implicitly used ExpyXB
    
    std::vector<std::size_t> FullOrder = this->Order; // never used in LR
    this->Order.resize(std::min((int) (n_nonzero(this->B) + this->ScreenSize + this->NoSelectK), (int)(this->p)));
    
    for (std::size_t t = 0; t < this->MaxIters; ++t) {
        this->Bprev = this->B;
        
        // Update the intercept
        if (this->intercept){
            const double b0old = this->b0;
            // const double partial_b0 = - arma::sum( *(this->y) / (1 + ExpyXB) );
            const double partial_b0 = - arma::dot( *(this->y) , 1/(1 + ExpyXB) );
            this->b0 -= partial_b0 / (this->n * LipschitzConst); // intercept is not regularized
            ExpyXB %= arma::exp( (this->b0 - b0old) * *(this->y));
        }
        
        for (auto& i : this->Order) {
            this->UpdateBiWithBounds(i);
        }
        
        this->RestrictSupport();
        
        // only way to terminate is by (i) converging on active set and (ii) CWMinCheck
        if (this->isConverged() && this->CWMinCheckWithBounds()) {
            break;
        }
    }
    
    this->result.Objective = this->objective;
    this->result.B = this->B;
    this->result.Model = this;
    this->result.b0 = this->b0;
    this->result.ExpyXB = ExpyXB;
    this->result.IterNum = this->CurrentIters;
    
    return this->result;
}

template class CDL012Exponential<arma::mat>;
template class CDL012Exponential<arma::sp_mat>;


#endif
