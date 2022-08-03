#include <iostream>
#include "gmdh.h"
#include <armadillo>


int main() {
    arma::mat x = { {1, 2, 3},
                    {4, 5, 6} };

    arma::vec y = { 0, 2 };


    GMDH::COMBI combi;
    combi.fit(x, y, GMDH::RegularityCriterion(0.5));

    (std::cin).get();

    return 0;
}