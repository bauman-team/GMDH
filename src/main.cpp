#include <iostream>
#include "gmdh.h"
#define print(x) std::cout<<x;


int main() {

    using namespace Eigen;
    VectorXd x(10); x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    int lags = 5;
    double validate_size = 0.33;
    std::pair<MatrixXd, VectorXd> ts = GMDH::convertToTimeSeries(x, lags);
    GMDH::splitted_data data = GMDH::splitTsData(ts.first, ts.second, validate_size);
    print(data.x_train << "\n\n");
    print(data.x_test << "\n\n");
    print(data.y_train << "\n\n");
    print(data.y_test << "\n\n");

    double test_size = 0.33;

    print("Original time series:\n" << x << "\n\n");

    GMDH::COMBI combi;
    combi.fit(data.x_train, data.y_train, GMDH::RegularityCriterionTS(test_size));

    std::cout << "The best polynom:\n" << combi.getBestPolymon() << std::endl;

    auto res = combi.predict(data.x_test);
    combi.save("model1.txt");
    combi.load("model1.txt");
    auto res2 = combi.predict(data.x_test);

    print("Predicted values before model saving:\n" << res << "\n\n");
    print("Predicted values after model loading:\n" << res2 << "\n\n");

    (std::cin).get();

    return 0;
}