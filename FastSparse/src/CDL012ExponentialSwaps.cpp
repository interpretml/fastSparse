#include "CDL012ExponentialSwaps.h"

template <class T>
CDL012ExponentialSwaps<T>::CDL012ExponentialSwaps(const T& Xi, const arma::vec& yi, const Params<T>& Pi) : CDSwaps<T>(Xi, yi, Pi) {
    twolambda2 = 2 * this->lambda2;
    qp2lamda2 = (LipschitzConst + twolambda2); // this is the univariate lipschitz const of the differentiable objective
    this->thr2 = (2 * this->lambda0) / qp2lamda2;
    this->thr = std::sqrt(this->thr2);
    stl0Lc = std::sqrt((2 * this->lambda0) * qp2lamda2);
    lambda1ol = this->lambda1 / qp2lamda2;
    Xy = Pi.Xy;

    this->Xy_neg_indices = Pi.Xy_neg_indices;
}

template <class T>
FitResult<T> CDL012ExponentialSwaps<T>::_FitWithBounds() {
    throw "This Error should not happen. Please report it as an issue to https://github.com/jiachangliu/fastSparse ";
}

template <class T>
FitResult<T> CDL012ExponentialSwaps<T>::_Fit() {
    // std::cout << "CDL012LogisticSwaps.cpp i'm in line 22\n";

    auto result = CDL012Exponential<T>(*(this->X), *(this->y), this->P).Fit(); // result will be maintained till the end
    this->b0 = result.b0; // Initialize from previous later....!
    this->B = result.B;
    arma::vec inverse_ExpyXB = result.inverse_ExpyXB; // Maintained throughout the algorithm
    
    double Fmin = arma::sum(inverse_ExpyXB);
    std::size_t maxindex;
    double Bmaxindex;
    
    this->P.Init = 'u';
    
    bool foundbetter = false;
    bool foundbetter_i = false;
    
    int start_NnzIndices_value = 0;
    arma::rowvec frequency_count = arma::zeros<arma::rowvec>(this->p);

    std::size_t ll_max = std::min(50, (int) this->p); // consider set 50 to be a parameter
    arma::uvec support_indices = arma::regspace<arma::uvec>(0, this->p - 1);
    for (std::size_t t = 0; t < this->MaxNumSwaps; ++t) {
        // std::cout << "Exponential Swaps is fitting!\n";
        ///////////////////////////////////////////////////// Sequential order
        // std::vector<std::size_t> NnzIndices = nnzIndicies(this->B, this->NoSelectK);

        ///////////////////////////////////////////////////// Priority Queue order
        std::vector<std::size_t> NnzIndices = nnzIndicies(this->B, this->NoSelectK);
        std::sort(NnzIndices.begin(), NnzIndices.end(), [&](std::size_t i, std::size_t j){
            if (frequency_count(i) == frequency_count(j)){
                return i < j;
            }
            return frequency_count(i) < frequency_count(j);
        });
        ///////////////////////////////////////////////////// 

        
        foundbetter = false;
        
        // TODO: Check if this should be Templated Operation
        arma::mat inverse_ExpyXBnojs = arma::zeros(this->n, NnzIndices.size());
        
        int j_index = -1;
        for (auto& j : NnzIndices)
        {
            // Remove NnzIndices[j]
            ++j_index;
            inverse_ExpyXBnojs.col(j_index) = inverse_ExpyXB % arma::exp( this->B.at(j) * matrix_column_get(*(this->Xy), j));

        }
        arma::mat gradients = - inverse_ExpyXBnojs.t() * *Xy;
        arma::mat abs_gradients = arma::abs(gradients);
        
        j_index = -1;
        for (auto& j : NnzIndices) {
            // std::cout << "trying to find alternative to index " + std::to_string(j) + "\n";
            // Set B[j] = 0
            ++j_index;
            arma::vec inverse_ExpyXBnoj = inverse_ExpyXBnojs.col(j_index);
            arma::rowvec gradient = gradients.row(j_index);
            arma::rowvec abs_gradient = abs_gradients.row(j_index);
            
            // arma::uvec indices = arma::sort_index(arma::abs(gradient), "descend");
            std::partial_sort(support_indices.begin(), support_indices.begin()+ll_max, support_indices.end(), [&](std::size_t ii, std::size_t jj){
                return abs_gradient(ii) > abs_gradient(jj);
            }); // partial sort
            foundbetter_i = false;
            
            double loss_noj = arma::sum(inverse_ExpyXBnoj);
            // TODO: make sure this scans at least 100 coordinates from outside supp (now it does not)
            for(std::size_t ll = 0; ll < ll_max; ++ll) {
                std::size_t i = support_indices(ll);
                
                if(this->B[i] == 0 && i >= this->NoSelectK) {
                    // Do not swap B[i] if i between 0 and NoSelectK;
                    
                    indices = (*(this->Xy_neg_indices))[i];
                    double d_minus_tmp = arma::sum(inverse_ExpyXBnoj.elem(indices))/loss_noj;

                    double ObjTemp = loss_noj * 2 * std::sqrt(d_minus_tmp*(1-d_minus_tmp));

                    // double Biold = 0;
                    // double partial_i = -0.5*std::log((1.0-d_minus_tmp)/d_minus_tmp);
                    // double Binew = Biold - partial_i;
                    double Binew = 0.5*std::log((1.0-d_minus_tmp)/d_minus_tmp);


//////////////////////////////////////////////////////////////////////////////////

                    if (ObjTemp < Fmin) {
                        // std::cout << "because loss terms are " + std::to_string(ObjTemp) + " vs " + std::to_string(Fmin) + "\n";
                        Fmin = ObjTemp;
                        maxindex = i;
                        Bmaxindex = Binew;
                        foundbetter_i = true;
                        // std::cout << "successfully swapping index " + std::to_string(j) + " with index " + std::to_string(i) + " at iteration " + std::to_string(ll) + "\n";
                    }  else {
                        // std::cout << "unsuccessfully swapping with index " + std::to_string(i) + "\n";
                        frequency_count(j) += 1;
                    }
                }
                
                if (foundbetter_i) {
                    this->B[j] = 0;
                    this->B[maxindex] = Bmaxindex;
                    this->P.InitialSol = &(this->B);
                    
                    // TODO: Check if this line is necessary. P should already have b0.
                    this->P.b0 = this->b0;
                    
                    result = CDL012Exponential<T>(*(this->X), *(this->y), this->P).Fit();
                    
                    inverse_ExpyXB = result.inverse_ExpyXB;
                    this->B = result.B;
                    this->b0 = result.b0;

                    // // early stopping
                    // const arma::ucolvec wronglyClassifiedIndicies = arma::find(inverse_ExpyXB > 1);
                    // if (wronglyClassifiedIndicies.size() == 0) {
                    //     // std::cout << "Early stopping because classification accuracy is 100%\n";
                    //     return result;
                    // }
                    // std::cout << "there are " + std::to_string(wronglyClassifiedIndicies.size()) + " wrongly classified points\n";

                    Fmin = arma::sum(inverse_ExpyXB);
                    foundbetter = true;
                    break;
                }
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

template class CDL012ExponentialSwaps<arma::mat>;
template class CDL012ExponentialSwaps<arma::sp_mat>;
