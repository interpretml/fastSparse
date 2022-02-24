#ifndef CDL012LogisticSwaps_H
#define CDL012LogisticSwaps_H
#include "RcppArmadillo.h"
#include "CD.h"
#include "CDSwaps.h"
#include "CDL012Logistic.h"
#include "FitResult.h"
#include "Params.h"
#include "utils.h"
#include "BetaVector.h"

template <class T>
class CDL012LogisticSwaps : public CDSwaps<T> {
    private:
        const double LipschitzConst = 0.25;
        double twolambda2;
        double qp2lamda2;
        double lambda1ol;
        double stl0Lc;
        arma::vec ExpyXB;
        // std::vector<double> * Xtr;
        T * Xy;

        double Fmin;

        // my new attributes
        arma::rowvec frequency_count;
        beta_vector Btemp;

        arma::vec ExpyXBnoji_mid;
        double partial_i_mid;
        arma::vec ExpyXBnoji_double;
        double partial_i_double;
        arma::vec ExpyXBnoji_triple;
        double partial_i_triple;

        arma::vec ExpyXBnoji;
        double Biold;
        double partial_i;
        double Binew;

    public:
        CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& P);

        FitResult<T> _FitWithBounds() final;
        
        FitResult<T> _Fit() final;

        inline double Objective(const arma::vec & r, const beta_vector & B) final;
        
        inline double Objective() final;

        FitResult<T> finetune();
        FitResult<T> replace_indexJ_with_indexI_and_finetune(std::size_t j, std::size_t i, double coef_i);
        bool evaluate_pruning_by_quadratic_cut_1point(double f1, double df1, double bestf);
        bool evaluate_pruning_by_quadratic_cut_2points(double f1, double x1, double df1, double f2, double x2, double df2, double bestf);
        bool evaluate_pruning_by_linear_cut_2points(double f1, double x1, double df1, double f2, double x2, double df2, double bestf);
        double one_gradientDescent_step(double Biold, double partial_i);

        void update_ExpyXB_and_partial(arma::vec & oldExpyXB, double BiOld, double BiNew, std::size_t i, arma::vec & newExpyXB, double & partial_i_new);
        void update_Biold_and_Binew(double & Biold, double & Binew, double partial_i);
        inline double Objective(const arma::vec & r, beta_vector & B, std::size_t i, double Binew);

        bool evaluate_early_break(std::size_t i);

};

template <class T>
inline double CDL012LogisticSwaps<T>::Objective(const arma::vec & r, beta_vector & B, std::size_t i, double Binew) {
    B[i] = Binew;
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->lambda0 * n_nonzero(B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012LogisticSwaps<T>::Objective(const arma::vec & r, const beta_vector & B) {
    auto l2norm = arma::norm(B, 2);
    return arma::sum(arma::log(1 + 1 / r)) + this->lambda0 * n_nonzero(B) + this->lambda1 * arma::norm(B, 1) + this->lambda2 * l2norm * l2norm;
}

template <class T>
inline double CDL012LogisticSwaps<T>::Objective() {
    auto l2norm = arma::norm(this->B, 2);
    return arma::sum(arma::log(1 + 1 / ExpyXB)) + this->lambda0 * n_nonzero(this->B) + this->lambda1 * arma::norm(this->B, 1) + this->lambda2 * l2norm * l2norm;
}

#endif
