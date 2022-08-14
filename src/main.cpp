#include <iostream>
#include "multi.h"


int main() {

    using namespace Eigen;

    std::ifstream dataStream;
    dataStream.open("../examples/Sber.csv");
    std::string dataLine;
    std::vector<double> dataValues;
    if (dataStream.is_open()) {
        while (std::getline(dataStream, dataLine)) {
            dataValues.push_back(std::atof(dataLine.c_str()));
        }
    }

    VectorXd data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size() - 50000);
    int lags = 10;
    double validateSize = 0.2;
    double testSize = 0.33;
    std::pair<MatrixXd, VectorXd> timeSeries = GMDH::convertToTimeSeries(data, lags);
    GMDH::SplittedData splittedData = GMDH::splitTimeSeries(timeSeries.first, timeSeries.second, validateSize);
    //std::cout << data.x_train << "\n\n";
    //std::cout << data.x_test << "\n\n";
    //std::cout << data.y_train << "\n\n";
    //std::cout << data.y_test << "\n\n";


    //std::cout << "Original time series:\n" << x << "\n\n";

    GMDH::MULTI multi;
    multi.fit(splittedData.xTrain, splittedData.yTrain, GMDH::RegularityCriterionTS(testSize, GMDH::Solver::fast), 3, 1);

    std::cout << "The best polynom:\n" << multi.getBestPolynomial() << std::endl;

    auto res = multi.predict(splittedData.xTest);
    multi.save("model1.txt");
    multi.load("model1.txt");
    auto res2 = multi.predict(splittedData.xTest);
    std::cout << "The best polynom after loading:\n" << multi.getBestPolynomial() << std::endl;

    //std::cout << "Predicted values before model saving:\n" << res << "\n\n";
    //std::cout << "Predicted values after model loading:\n" << res2 << "\n\n";

    //(std::cin).get();

    return 0;
}