#include <iostream>
#include <combi.h>
#include <multi.h>
#include <ria.h>


int main() {

    using namespace Eigen;

    /*std::ifstream dataStream;
    std::string gmdhDir = std::getenv("GMDH_ROOT");
    dataStream.open(gmdhDir + "/examples/Sber.csv");
    std::string dataLine;
    std::vector<double> dataValues;
    VectorXd data;
    if (dataStream.is_open()) {
        while (std::getline(dataStream, dataLine)) {
            dataValues.push_back(std::atof(dataLine.c_str()));
        }
        data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size() - 50000);
        dataStream.close();
    }*/

    VectorXd data(12); data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    std::cout << "Original data:\n" << data << "\n\n";

    int lags = 4;
    auto timeSeries = GMDH::timeSeriesTransformation(data, lags);
    std::cout << "X data:\n" << timeSeries.first << "\n\n";
    std::cout << "Y data:\n" << timeSeries.second << "\n\n";

    double validateSize = 0.33;
    GMDH::SplittedData splittedData = GMDH::splitData(timeSeries.first, timeSeries.second, validateSize);

    std::cout << "X train data:\n" << splittedData.xTrain << "\n\n";
    std::cout << "X test data:\n" << splittedData.xTest << "\n\n";
    std::cout << "Y train data:\n" << splittedData.yTrain << "\n\n";
    std::cout << "Y test data:\n" << splittedData.yTest << "\n\n";
    std::cout << std::endl;

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

    double testSize = 0.33;
    int kBest = 10;
    int pAverage = 1;
    int threads = 1;
    int verbose = 1;
    double limit = 0;
    auto criterion = GMDH::Criterion(GMDH::CriterionType::stability);
    auto polynomialType = GMDH::PolynomialType::linear_cov;

    GMDH::MIA mia;
    mia.fit(splittedData.xTrain, splittedData.yTrain, criterion, 
            kBest, polynomialType, testSize, pAverage, threads, verbose, limit);

    std::cout << "\nThe best polynomial:\n" << mia.getBestPolynomial() << "\n\n";
    
    int lagsToPredict = 3;
    auto res = mia.predict(splittedData.xTest(0, all), lagsToPredict);
    std::cout << "Y predicted data:\n" << res << "\n\n";

    mia.save("mia_model.txt");

    if (!mia.load("mia_model.txt")) {
        auto res2 = mia.predict(splittedData.xTest(0, all), lagsToPredict);
        std::cout << "The best polynomial after model loading:\n" << mia.getBestPolynomial() << "\n\n";
        std::cout << "Y predicted data after model loading:\n" << res2 << "\n\n";
    }
    return 0;
}