#include "CDL012LogisticSwaps.h"

template <class T>
CDL012LogisticSwaps<T>::CDL012LogisticSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDSwaps<T>(Xi, yi, Pi) {
    // std::cout << "CDL012LogisticSwaps.cpp i'm in line 5\n";
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    stl0Lc = std::sqrt((2 * this->lambda0) * qp2lamda2);
    lambda1ol = this->lambda1 / qp2lamda2;
    Xy = Pi.Xy;

    frequency_count = arma::zeros<arma::rowvec>(this->p);
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::_FitWithBounds() {
    throw "This Error should not happen. Please report it as an issue to https://github.com/jiachangliu/fastSparse";
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::_Fit() {
    // std::cout << "CDL012LogisticSwaps.cpp i'm in line 22\n";
    auto result = finetune(); // start result from a faster algorithm; result will be maintained till the end 
    
    this->P.Init = 'u'; // prevent initialization in all 0's for the coefficients in CDL012Exponential.h
    
    int start_NnzIndices_value = 0;

    // TODO: make sure this scans at least 100 coordinates from outside supp (now it does not)
    std::size_t ll_max = std::min(50, (int) this->p); // consider set 50 to be a parameter
    arma::uvec indices = arma::regspace<arma::uvec>(0, this->p - 1);

    for (std::size_t t = 0; t < this->MaxNumSwaps; ++t) {
        ///////////////////////////////////////////////////// Priority Queue order
        std::vector<std::size_t> NnzIndices = nnzIndicies(this->B, this->NoSelectK);
        std::sort(NnzIndices.begin(), NnzIndices.end(), [&](std::size_t i, std::size_t j){
            if (frequency_count(i) == frequency_count(j)){
                return i < j;
            }
            return frequency_count(i) < frequency_count(j);
        });
        ///////////////////////////////////////////////////// 
        
        // TODO: Check if this should be Templated Operation
        arma::mat ExpyXBnojs = arma::zeros(this->n, NnzIndices.size());
        
        int j_index = -1;
        for (auto& j : NnzIndices)
        {
            // Remove NnzIndices[j]
            ++j_index;
            ExpyXBnojs.col(j_index) = this->ExpyXB % arma::exp( - this->B.at(j) * matrix_column_get(*(this->Xy), j));

        }
        arma::mat gradients = - 1/(1 + ExpyXBnojs).t() * *Xy;
        arma::mat abs_gradients = arma::abs(gradients);
        
        j_index = -1;
        bool foundbetter = false;

        for (auto& j : NnzIndices) {
        
        // for (j_index=0; j_index < NnzIndices.size(); j_index++) {
        //     j = NnzIndices(j_index);

            // Set B[j] = 0
            ++j_index;
            arma::vec ExpyXBnoj = ExpyXBnojs.col(j_index);
            arma::rowvec gradient = gradients.row(j_index);
            arma::rowvec abs_gradient = abs_gradients.row(j_index);
            
            // arma::uvec indices = arma::sort_index(abs_gradient, "descend");
            // ToDo: change ll_max to number of alternative features to consider
            std::partial_sort(indices.begin(), indices.begin()+ll_max, indices.end(), [&](std::size_t ii, std::size_t jj){
                return abs_gradient(ii) > abs_gradient(jj);
            });

            Btemp = this->B;
            Btemp[j] = 0.0;
            double ObjTemp = Objective(ExpyXBnoj, Btemp);
            if (ObjTemp < this->Fmin) {
                result = replace_indexJ_with_indexI_and_finetune(j, j, 0);
                foundbetter = true;
                break;
            }
            
            std::size_t previous_i = -1; // index of the largest gradient
            for(std::size_t ll = 0; ll < ll_max; ++ll) {
                std::size_t i = indices(ll);
                
                if(this->B[i] != 0 || i < this->NoSelectK) {
                    // Do not swap B[i] if i between 0 and NoSelectK;
                    continue;
                }

                ExpyXBnoji = ExpyXBnoj;
                
                Biold = 0;
                partial_i = gradient[i];
                
                if (previous_i != -1) {
                    Btemp[previous_i] = 0.0;
                }
                previous_i = i;
                
                // BELOW NEEDs REVISION
                Binew = one_gradientDescent_step(Biold, partial_i);
                // double Binew = clamp(std::copysign(z, x), this->Lows[i], this->Highs[i]); // no need to check if >= sqrt(2lambda_0/Lc)

///////////////////////////////////////////////////////////// my implementation
                bool early_break = evaluate_pruning_by_quadratic_cut_1point(ObjTemp+this->lambda0, partial_i, this->Fmin);
                if (early_break){
                    continue;
                }

                early_break = evaluate_early_break(i);
                if (early_break){
                    frequency_count(j) += 1;
                    continue;
                }

                for(std::size_t innerindex=1; innerindex < 10; ++innerindex) {
                    update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew, i, ExpyXBnoji, partial_i);
                    bool converged = std::abs((Binew - Biold)/Biold) < 0.0001;
                    update_Biold_and_Binew(Biold, Binew, partial_i);
                    if (converged) {
                        break;
                    }
                }
                
                ExpyXBnoji %= arma::exp( (Binew - Biold) *  matrix_column_get(*Xy, i));
                ObjTemp = Objective(ExpyXBnoji, Btemp, i, Binew);
//////////////////////////////////////////////////////////////////////////////////

                if (ObjTemp >= this->Fmin) {
                    frequency_count(j) += 1;
                    continue;
                }

                // std::cout << "because loss terms are " + std::to_string(ObjTemp) + " vs " + std::to_string(this->Fmin) + "\n";
                // std::cout << "successfully swapping index " + std::to_string(j) + " with index " + std::to_string(i) + " at iteration " + std::to_string(ll) + "\n"; // mine
                this->Fmin = ObjTemp;
                result = replace_indexJ_with_indexI_and_finetune(j, i, Binew);
                foundbetter = true;
                break;

            }
            
            //auto end2 = std::chrono::high_resolution_clock::now();
            //std::cout<<"restricted:  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms " << std::endl;
            
            if (foundbetter){
                break;
            } 

        }
        
        if(!foundbetter) {
            // Early exit to prevent looping
            return result;
        }
    }
    
    //result.Model = this;
    return result;
}

template <class T>
bool CDL012LogisticSwaps<T>::evaluate_early_break(std::size_t i){
    bool early_break;
    double Binew_double = Biold - 2*partial_i/qp2lamda2;
    update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew_double, i, ExpyXBnoji_double, partial_i_double);

    // High-level Comments:
    // We first decide whether B_i* is within [Binew, Binew_double] or [Binew_double, Binew_triple]
    // Although the methods below share many similarities, there are some subtle implementation differences between these two intervals.
    // If B_i* is within [Binew, Binew_double], we do a binary search to cut the interval in half, [Binew, Bimid] and [Bimid, Binew_double]
    //          we build cuting planes (curves) in [Binew, Bimid] or [Bimid, Binew_double]
    // If B_i* is within [Binew_double, Binew_triple], we don't do binary search as this costs more computation.
    //          however, if B_i* is beyond Binew_triple, we calculate another lower bound under this case
    if (partial_i * partial_i_double < 0) { // B_i* is between Binew and Binew_double
        double Binew_mid = (Binew + Binew_double)/2;
        update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew_mid, i, ExpyXBnoji_mid, partial_i_mid);
        double objective_mid = Objective(ExpyXBnoji_mid, Btemp, i, Binew_mid);

        early_break = evaluate_pruning_by_quadratic_cut_1point(objective_mid, partial_i_mid, this->Fmin);
        if (early_break){
            return true;
        }

        if (partial_i_mid * partial_i < 0) { // B_i* is between Binew and Binew_mid
            update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew, i, ExpyXBnoji, partial_i);
            double objective_new = Objective(ExpyXBnoji, Btemp, i, Binew);
            // early_break = evaluate_pruning_by_linear_cut_2points(objective_new, Binew, partial_i, objective_mid, Binew_mid, partial_i_mid, this->Fmin);
            early_break = evaluate_pruning_by_quadratic_cut_2points(objective_new, Binew, partial_i, objective_mid, Binew_mid, partial_i_mid, this->Fmin);
            if (early_break) {
                return true;
            }
            update_Biold_and_Binew(Biold, Binew, partial_i);
        } else { // B_i* is between Binew_mid and Binew_double
            double objective_double = Objective(ExpyXBnoji_double, Btemp, i, Binew_double);
            // early_break = evaluate_pruning_by_linear_cut_2points(objective_double, Binew_double, partial_i_double, objective_mid, Binew_mid, partial_i_mid, this->Fmin);
            early_break = evaluate_pruning_by_quadratic_cut_2points(objective_double, Binew_double, partial_i_double, objective_mid, Binew_mid, partial_i_mid, this->Fmin);
            if (early_break) {
                return true;
            }
            update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew, i, ExpyXBnoji, partial_i);
            update_Biold_and_Binew(Biold, Binew, partial_i);
        }
    } else { // Binew and Binew_double are on the same side of B_i*
        double objective_double = Objective(ExpyXBnoji_double, Btemp, i, Binew_double);
        bool early_break = evaluate_pruning_by_quadratic_cut_1point(objective_double, partial_i_double, this->Fmin);
        if (early_break){
            return true;
        }
        double Binew_triple = Biold - partial_i/qp2lamda2*3;
        update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew_triple, i, ExpyXBnoji_triple, partial_i_triple);
        if (partial_i_triple * partial_i_double < 0) { // B_i* is between Binew_double and Binew_triple
            double objective_triple = Objective(ExpyXBnoji_triple, Btemp, i, Binew_triple);
            // early_break = evaluate_pruning_by_linear_cut_2points(objective_double, Binew_double, partial_i_double, objective_triple, Binew_triple, partial_i_triple, this->Fmin);
            early_break = evaluate_pruning_by_quadratic_cut_2points(objective_double, Binew_double, partial_i_double, objective_triple, Binew_triple, partial_i_triple, this->Fmin);
            if (early_break) {
                return true;
            }
        } else {
            double objective_triple = Objective(ExpyXBnoji_triple, Btemp, i, Binew_triple);
            early_break = evaluate_pruning_by_quadratic_cut_1point(objective_triple, partial_i_triple, this->Fmin);
            if (early_break){
                // std::cout << "triple strongly convex lower bound is working!!!\n";
                return true;
            }
            // else {
                // std::cout << "optimal value goes beyond triple distance!\n";
                // exit(0);
            // }
        }
        update_ExpyXB_and_partial(ExpyXBnoji, Biold, Binew, i, ExpyXBnoji, partial_i);
        update_Biold_and_Binew(Biold, Binew, partial_i);
    }
    return false;
}

template <class T>
void CDL012LogisticSwaps<T>::update_ExpyXB_and_partial(arma::vec & oldExpyXB, double BiOld, double BiNew, std::size_t i, arma::vec & newExpyXB, double & partial_i_new){
    double BiDiff = BiNew - BiOld;
    arma::vec Xy_i = matrix_column_get(*Xy, i);
    newExpyXB = oldExpyXB % arma::exp( BiDiff * Xy_i);
    partial_i_new = -arma::dot( Xy_i, 1/(1+newExpyXB) ) + twolambda2 * BiNew;
}

template <class T>
void CDL012LogisticSwaps<T>::update_Biold_and_Binew(double & Biold, double & Binew, double partial_i){
    Biold = Binew;
    Binew = one_gradientDescent_step(Biold, partial_i);
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::finetune(){
    auto result = CDL012Logistic<T>(*(this->X), *(this->y), this->P).Fit();
    this->ExpyXB = result.ExpyXB;
    this->B = result.B;
    this->b0 = result.b0;
    this->Fmin = result.Objective;
    return result;
}

template <class T>
FitResult<T> CDL012LogisticSwaps<T>::replace_indexJ_with_indexI_and_finetune(std::size_t j, std::size_t i, double coef_i) {
    this->B[j] = 0;
    this->B[i] = coef_i;
    this->P.InitialSol = &(this->B);
    // TODO: Check if this line is necessary. P should already have b0.
    this->P.b0 = this->b0;
    // auto result = CDL012Logistic<T>(*(this->X), *(this->y), this->P).Fit();
    // this->ExpyXB = result.ExpyXB;
    // this->B = result.B;
    // this->b0 = result.b0;
    // this->Fmin = result.Objective;
    auto result = finetune();
    return result;
}

//ToDo: allowQuadCut1
template <class T>
bool CDL012LogisticSwaps<T>::evaluate_pruning_by_quadratic_cut_1point(double f1, double df1, double bestf){
    // f1 -> current loss, x1 -> current variable, df1 -> current derivative
    // return true if lowerBound >
    double lowerBound = f1 - (df1*df1)/(4*this->lambda2);
    if (lowerBound > bestf){
        return true;
    }
    return false;
}

template <class T>
bool CDL012LogisticSwaps<T>::evaluate_pruning_by_quadratic_cut_2points(double f1, double x1, double df1, double f2, double x2, double df2, double bestf){
    double x_intersect = -(f1 - df1*x1 + this->lambda2*(x1 - x2)*(x1 + x2) - f2 + df2*x2)/(df1 - df2 - 2*this->lambda2*(x1 - x2));
    double tmp = x_intersect - x2;
    double lowerBound = f2 + df2*tmp + this->lambda2 * tmp*tmp;
    if (lowerBound > bestf){
        return true;
    }
    return false;
}

template <class T>
bool CDL012LogisticSwaps<T>::evaluate_pruning_by_linear_cut_2points(double f1, double x1, double df1, double f2, double x2, double df2, double bestf){
    double lowerBound = (df1*df2 *(x1 -x2) - f1*df2 + f2*df1) / (df1 - df2);
    if (lowerBound > bestf){
        return true;
    }
    return false;
}

template <class T>
double CDL012LogisticSwaps<T>::one_gradientDescent_step(double Biold, double partial_i){
    double x = Biold - partial_i/this->qp2lamda2;
    double z = std::abs(x) - this->lambda1ol;
    double Binew = std::copysign(z, x);
    return Binew;
}

template class CDL012LogisticSwaps<arma::mat>;
template class CDL012LogisticSwaps<arma::sp_mat>;
