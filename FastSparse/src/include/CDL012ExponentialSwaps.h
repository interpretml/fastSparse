#ifndef CDL012ExponentialSwaps_H
#define CDL012ExponentialSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012Exponential.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012ExponentialSwaps : public CDSwaps<T> {
    private:
        const double LipschitzConst = 0.25;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        double stl0Lc;
        arma::vec ExpyXB;
        // std::vector<double> * Xtr;
        T * Xy;

        arma::uvec indices;

        //////// new variables for expo loss
        arma::vec inverse_ExpyXB;
        double d_minus;
        double d_plus;
        double current_expo_loss;

        std::unordered_map<std::size_t, arma::uvec> * Xy_neg_indices;

    public:
        CDL012ExponentialSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;

};


template <class T>
inline double CDL012ExponentialSwaps<T>::Objective(const arma::vec & r, const beta_vector & B) {
    // auto l2norm = arma::norm(B, 2);
    return arma::sum(r) + this->lambda0 * n_nonzero(B); //+ this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012ExponentialSwaps<T>::Objective() {
    // auto l2norm = arma::norm(this->B, 2);
    return arma::sum(this->inverse_ExpyXB) + this->lambda0 * n_nonzero(this->B); //+ this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

#endif
