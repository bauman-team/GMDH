#include <gtest/gtest.h>
#include <combi.h>


TEST(save_load_test, trained_model)
{
/* MOVE TO FIXTURES */
    using namespace Eigen;

    std::vector<double> dataValues{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    VectorXd data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
    int lags = 5;
    double validateSize = 0.2;
    double testSize = 0.33;
    auto timeSeries = GMDH::convertToTimeSeries(data, lags);
    GMDH::SplittedData splittedData = GMDH::splitTimeSeries(timeSeries.first, timeSeries.second, validateSize);

    GMDH::COMBI combi;
    combi.fit(splittedData.xTrain, splittedData.yTrain, GMDH::RegularityCriterionTS(testSize, GMDH::Solver::fast), 1, 1);

/* MOVE TO FIXTURES */

    auto res = combi.getBestPolynomial(); // TODO: delete dependencies (getBestPolynomial)
    combi.save("model1.txt");
    combi.load("model1.txt");
    auto res2 = combi.getBestPolynomial();

    ASSERT_STREQ(res.c_str(), res2.c_str());
}
