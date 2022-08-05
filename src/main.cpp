#include <iostream>
#include "gmdh.h"
#include <armadillo>


int main() {
    arma::mat x = { {1, 2, 3},
                    {4, 5, 6} };

    arma::vec y = { 0, 2 };


    GMDH::COMBI combi;
    combi.fit(x, y, GMDH::RegularityCriterion(0.5, true, 2));
    double res1 = combi.predict(arma::rowvec(x.n_cols, arma::fill::randu));
    arma::vec res2 = combi.predict(arma::mat(2, x.n_cols, arma::fill::randu));
    combi.save("model1.txt");
    combi.load("model1.txt");
    arma::vec res3 = combi.predict(arma::mat(2, x.n_cols, arma::fill::randu));

    (std::cin).get();

    return 0;
}