#include <iostream>
#include "gmdh.h"
#include <armadillo>


int main() {

    arma::vec x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int lags = 5;
    double validate_size = 0.33;
    double test_size = 0.5;

    x.print("Original time series: ");

    std::pair<arma::mat, arma::vec> ts = GMDH::convertToTimeSeries(x, lags);
    GMDH::splitted_data data = GMDH::splitTsData(ts.first, ts.second, validate_size);

    data.x_train.print("x_train: ");
    data.y_train.print("y_train: ");
    data.x_test.print("x_test: ");
    data.y_test.print("y_test: ");

    GMDH::COMBI combi;
    combi.fit(data.x_train, data.y_train, GMDH::RegularityCriterionTS(test_size));

    std::cout << "The best polynom:\n\  " << combi.getBestPolymon() << std::endl;

    arma::vec res = combi.predict(data.x_test);
    combi.save("model1.txt");
    combi.load("model1.txt");
    arma::vec res2 = combi.predict(data.x_test);

    res.print("Predicted values before model saving: ");
    res2.print("Predicted values after model loading: ");

    (std::cin).get();

    return 0;
}