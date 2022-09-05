#include <iostream>
#include <combi.h>
#include <ria.h>


int main() {

    using namespace Eigen;

    std::ifstream dataStream;
    dataStream.open("../../examples/Sber.csv");
    std::string dataLine;
    std::vector<double> dataValues;
    VectorXd data;
    if (dataStream.is_open()) {
        while (std::getline(dataStream, dataLine)) {
            dataValues.push_back(std::atof(dataLine.c_str()));
        }
        data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size() - 50000);
        dataStream.close();
    }

    //VectorXd data(10); data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	
    int lags = 20;
    double validateSize = 0.2;
    double testSize = 0.33;
    auto timeSeries = GMDH::timeSeriesTransformation(data, lags);
    GMDH::SplittedData splittedData = GMDH::splitData(timeSeries.first, timeSeries.second, validateSize);

    /*std::cout << splittedData.xTrain << "\n\n";
    std::cout << splittedData.xTest << "\n\n";
    std::cout << splittedData.yTrain << "\n\n";
    std::cout << splittedData.yTest << "\n\n";*/

    /*
        x1    x2     x3     x4   => x1, x2
        1111

        1000, 0100

        1000 (1111 ^ 1000 = 0111)

        1100 !
        1010
        1001

        1000 + 0100 (1111 ^ 1100 = 0011)
 
        0110 !
        0101

        1100 (1111 ^ 1100 = 0011)

        1110
        1101

        1100 + 0110 (1111 ^ 1110 = 0001)

        0111


        x1 + x2 !
        x1 + x3
        x1 + x4
        x2 + x3 !
        x2 + x4

        x1 + x2 + x3 => (1, 2, 3)
        x1 + x2 + x4
        x2 + x3 + x1 (2, 3, 1) => (1, 2, 3) -
        x2 + x3 + x4

    */

    //std::cout << "Original time series:\n" << data << "\n\n";
    auto criterion = GMDH::Criterion(GMDH::CriterionType::regularity);
    GMDH::RIA mia;
    mia.fit(splittedData.xTrain, splittedData.yTrain, criterion, 3,
             GMDH::PolynomialType::quadratic, testSize, 0, 0, 2, 1, 1, 0);

    std::cout << "\nThe best polynomial:\n\n" << mia.getBestPolynomial() << std::endl;

    auto res = mia.predict(splittedData.xTest);
    mia.save("model1.txt");
    mia.load("model1.txt");
    auto res2 = mia.predict(splittedData.xTest);

    std::cout << "\nThe best polynomial after loading:\n\n" << mia.getBestPolynomial() << std::endl;

    /*for (int i = 0; i < 20; ++i)
        std::cout << splittedData.yTest[i] << " " << res[i] << " " << res2[i] << "\n";*/

    //std::cout << "Predicted values before model saving:\n" << res << "\n\n";
    //std::cout << "Predicted values after model loading:\n" << res2 << "\n\n";

    //(std::cin).get();

    GMDH::COMBI combi;
    combi.fit(splittedData.xTrain, splittedData.yTrain, criterion, testSize, false, 0, (uint8_t)1, -123, 1);//, 0, 0, 0, -123, 1);
    return 0;
}