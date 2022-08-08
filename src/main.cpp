#include <iostream>
#include "combi.h"

int main() {

    using namespace Eigen;

    std::ifstream o;
    o.open("../Sber.csv");
    std::string s;
    std::vector<double> v;
    if (o.is_open())
    {
        while (std::getline(o, s))
        {
            v.push_back(std::atof(s.c_str()));
        }
    }

    VectorXd x = Map<VectorXd, Unaligned>(v.data(), v.size() - 100000);
    int lags = 10;
    double validate_size = 0.2;
    double test_size = 0.33;
    std::pair<MatrixXd, VectorXd> ts = GMDH::convertToTimeSeries(x, lags);
    GMDH::splitted_data data = GMDH::splitTsData(ts.first, ts.second, validate_size);
    //std::cout << data.x_train << "\n\n";
    //std::cout << data.x_test << "\n\n";
    //std::cout << data.y_train << "\n\n";
    //std::cout << data.y_test << "\n\n";


    //std::cout << "Original time series:\n" << x << "\n\n";

    GMDH::COMBI combi;
    std::cout << boost::typeindex::type_id_with_cvr<decltype(combi)>();
    combi.fit(data.x_train, data.y_train, GMDH::RegularityCriterionTS(test_size));

    //std::cout << "The best polynom:\n" << combi.getBestPolymon() << std::endl;

    auto res = combi.predict(data.x_test);
    combi.save("model1.txt");
    combi.load("model1.txt");
    auto res2 = combi.predict(data.x_test);

    //std::cout << "Predicted values before model saving:\n" << res << "\n\n";
    //std::cout << "Predicted values after model loading:\n" << res2 << "\n\n";

    (std::cin).get();

    return 0;
}