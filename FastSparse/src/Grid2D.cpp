#include "Grid2D.h"

template <class T>
Grid2D<T>::Grid2D(const T& Xi, const arma::vec& yi, const GridParams<T>& PGi)
{
    // automatically selects lambda_0 (but assumes other lambdas are given in PG.P.ModelParams)
    X = &Xi;
    y = &yi;
    p = Xi.n_cols;
    PG = PGi;
    G_nrows = PG.G_nrows;
    G_ncols = PG.G_ncols;
    G.reserve(G_nrows);
    Lambda2Max = PG.Lambda2Max;
    Lambda2Min = PG.Lambda2Min;
    LambdaMinFactor = PG.LambdaMinFactor;
    
    P = PG.P;
}

template <class T>
Grid2D<T>::~Grid2D(){
    delete Xtr;
    if (PG.P.Specs.Logistic) {
        delete PG.P.Xy;
    }
    if (PG.P.Specs.SquaredHinge) {
        delete PG.P.Xy;
    }
    if (PG.P.Specs.Exponential) {
        delete PG.P.Xy;
        delete PG.P.Xy_neg_indices;
    }
}

template <class T>
std::vector< std::vector<std::unique_ptr<FitResult<T>> > > Grid2D<T>::Fit() {
    arma::vec Xtrarma;
    
    if (PG.P.Specs.Logistic) {
        // std::cout << "Grid2D.cpp i'm in line 35\n";
        auto n = X->n_rows;
        double b0 = 0;
        arma::vec ExpyXB =  arma::ones<arma::vec>(n);
        if (PG.intercept) {
            for (std::size_t t = 0; t < 50; ++t) {
                double partial_b0 = - arma::sum( *y / (1 + ExpyXB) );
                b0 -= partial_b0 / (n * 0.25); // intercept is not regularized
                ExpyXB = arma::exp(b0 * *y);
            }
        }
        PG.P.b0 = b0;
        Xtrarma = arma::abs(- arma::trans(*y /(1+ExpyXB)) * *X).t(); // = gradient of logistic loss at zero
        //Xtrarma = 0.5 * arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero
        
        T Xy =  matrix_vector_schur_product(*X, y); // X->each_col() % *y;
        
        PG.P.Xy = new T;
        *PG.P.Xy = Xy;
    } else if (PG.P.Specs.Exponential) {
        // std::cout << "Grid2D.cpp i'm in line 61\n";
        auto n = X->n_rows;
        double b0 = 0;
        // arma::vec ExpyXB =  arma::ones<arma::vec>(n);
        // if (PG.intercept) {
        //     for (std::size_t t = 0; t < 50; ++t) {
        //         double partial_b0 = - arma::sum( *y / (1 + ExpyXB) );
        //         b0 -= partial_b0 / (n * 0.25); // intercept is not regularized
        //         ExpyXB = arma::exp(b0 * *y);
        //     }
        // }

        
        // PG.P.b0 = b0;
        // Xtrarma = arma::abs(- arma::trans(*y /(1+ExpyXB)) * *X).t(); // = gradient of logistic loss at zero
        // //Xtrarma = 0.5 * arma::abs(y->t() * *X).t(); // = gradient of logistic loss at zero

        T Xy =  matrix_vector_schur_product(*X, y); // X->each_col() % *y;
        
        PG.P.Xy = new T;
        *PG.P.Xy = Xy;

        std::unordered_map<std::size_t, arma::uvec> Xy_neg_indices;
        for (size_t tmp = 0; tmp < Xy.n_cols; ++tmp){
            Xy_neg_indices.insert(std::make_pair(tmp, arma::find(matrix_column_get(*(PG.P.Xy), tmp) < 0)));
        }
        Xy_neg_indices.insert(std::make_pair(-1, arma::find(*y < 0)));
        PG.P.Xy_neg_indices = new std::unordered_map<std::size_t, arma::uvec>;
        *PG.P.Xy_neg_indices = Xy_neg_indices;

        // indices = (*(this->Xy_neg_indices))[-1];
        // // this->d_minus = arma::sum(this->inverse_ExpyXB.elem(indices)) / arma::sum(this->inverse_ExpyXB);
        // this->d_minus = arma::sum(this->inverse_ExpyXB.elem(indices)) / this->current_expo_loss;
        // const double partial_b0 = -0.5*std::log((1-this->d_minus)/this->d_minus);
        // this->b0 -= partial_b0;

        arma::vec inverse_ExpyXB =  arma::ones<arma::vec>(n);
        // calcualte the exponential intercept when all coordinates are zero
        b0 = 0.0;
        if (PG.intercept) {
            arma::uvec indices = Xy_neg_indices[-1];
            double d_minus = (double)indices.n_elem / (double)n;
            double partial_b0 = -0.5*std::log((1-d_minus)/d_minus);
            b0 -= partial_b0;
            inverse_ExpyXB %= arma::exp( partial_b0 * *y);
        }
        PG.P.b0 = b0;
        Xtrarma = arma::abs(- arma::trans(*y % inverse_ExpyXB) * *X).t(); // = gradient of logistic loss at zero
    } else if (PG.P.Specs.SquaredHinge) {
        auto n = X->n_rows;
        double b0 = 0;
        arma::vec onemyxb =  arma::ones<arma::vec>(n);
        arma::uvec indices = arma::find(onemyxb > 0);
        if (PG.intercept){
            for (std::size_t t = 0; t < 50; ++t){
                double partial_b0 = arma::sum(2 * onemyxb.elem(indices) % (- y->elem(indices) ) );
                b0 -= partial_b0 / (n * 2); // intercept is not regularized
                onemyxb = 1 - (*y * b0);
                indices = arma::find(onemyxb > 0);
            }
        }
        PG.P.b0 = b0;
        T indices_rows = matrix_rows_get(*X, indices);
        Xtrarma = 2 * arma::abs(arma::trans(y->elem(indices) % onemyxb.elem(indices))* indices_rows).t(); // = gradient of loss function at zero
        //Xtrarma = 2 * arma::abs(y->t() * *X).t(); // = gradient of loss function at zero
        T Xy =  matrix_vector_schur_product(*X, y); // X->each_col() % *y;
        PG.P.Xy = new T;
        *PG.P.Xy = Xy;
    } else {
        Xtrarma = arma::abs(y->t() * *X).t();
    }
    
    
    double ytXmax = arma::max(Xtrarma);
    
    std::size_t index;
    if (PG.P.Specs.L0L1) {
        index = 1;
        if(G_nrows != 1) {
            Lambda2Max = ytXmax;
            Lambda2Min = Lambda2Max * LambdaMinFactor;
        }
    } else if (PG.P.Specs.L0L2) {
        index = 2;
    }
    
    arma::vec Lambdas2 = arma::logspace(std::log10(Lambda2Min), std::log10(Lambda2Max), G_nrows);
    Lambdas2 = arma::flipud(Lambdas2);
    
    std::vector<double> Xtrvec = arma::conv_to< std::vector<double> >::from(Xtrarma);
    
    Xtr = new std::vector<double>(X->n_cols); // needed! careful
    
    
    PG.XtrAvailable = true;
    // Rcpp::Rcout << "Grid2D Start\n";
    for(std::size_t i=0; i<Lambdas2.size();++i) { //auto &l : Lambdas2
        // Rcpp::Rcout << "Grid1D Start: " << i << "\n";
        *Xtr = Xtrvec;
        
        PG.Xtr = Xtr;
        PG.ytXmax = ytXmax;
        
        PG.P.ModelParams[index] = Lambdas2[i];
        
        if (PG.LambdaU == true)
            PG.Lambdas = PG.LambdasGrid[i];
        
        //std::vector<std::unique_ptr<FitResult>> Gl();
        //auto Gl = Grid1D(*X, *y, PG).Fit();
        // Rcpp::Rcout << "Grid1D Start: " << i << "\n";
        G.push_back(std::move(Grid1D<T>(*X, *y, PG).Fit()));
    }
    
    return std::move(G);
    
}

template class Grid2D<arma::mat>;
template class Grid2D<arma::sp_mat>;
