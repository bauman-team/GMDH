#include <gtest/gtest.h>
#include <combi.h>
#include <iostream>

using namespace GMDH;
using namespace Eigen;

class TestCOMBI : public ::testing::Test
{
protected:
	void SetUp()
	{
        const int lags = 5; // TODO: NO CONST????
        const double testSize = 0.33;


        /* SET MODEL FOR PREDICT GROWING LINEAR DEPENDENCE TIME SERIES */
        std::vector<double> dataValues{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        VectorXd data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
        setModelTS(testGrowingLinearDependenceTS, std::move(data), lags, testSize, Solver::fast);


        /* SET MODEL FOR PREDICT DESCENDING LINEAR DEPENDENCE TIME SERIES */
        dataValues = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
        setModelTS(testDescendingLinearDependenceTS, std::move(data), lags, testSize, Solver::fast);


        /* SET MODEL FOR PREDICT GROWING LINEAR DEPENDENCE WITH NOISE TIME SERIES */
        dataValues = { 10, 19, 31, 40, 50, 62, 71, 79, 89, 98 };
        data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
        setModelTS(testGrowingLinearDependenceWithNoiseTS, std::move(data), lags, testSize, Solver::fast);


        /* SET MODEL FOR PREDICT LINEAR POLINOMIAL DEPENDENCE */


        /* SET MODEL FOR PREDICT LINEAR POLINOMIAL DEPENDENCE WITH NOISE */


        //data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
        
	}
	void TearDown()
	{
		// if it need clear memory
	}
    void setModelTS(COMBI& combi, VectorXd &&data, int lags, double testSize, Solver solverFunc) {
        auto timeSeries = convertToTimeSeries(data, lags);
        SplittedData splittedData = splitTimeSeries(timeSeries.first, timeSeries.second, 0);

        combi.fit(splittedData.xTrain, splittedData.yTrain, RegularityCriterionTS(testSize, solverFunc), 1, 1);
    }
    void setModel() {

    }
	COMBI testGrowingLinearDependenceTS, 
    testDescendingLinearDependenceTS,
    testGrowingLinearDependenceWithNoiseTS,
    testLinearPolinomailDependence, // TODO: polinomial tests
    testLinearPolinomailDependenceWithNoise;
};


::testing::AssertionResult PredictionEvaluation(VectorXd predict, VectorXd real, int precision) // TODO: check precision ????
{
    auto truncLast{ 10. };
    truncLast = std::pow(truncLast, precision);
    for (auto itPred = predict.begin(), itReal = real.begin(); itReal != real.end(); ++itPred, ++itReal) {
        if (static_cast<int64_t>(round(*itPred * truncLast)) != static_cast<int64_t>(round(*itReal * truncLast)))
		    return ::testing::AssertionFailure();
            //std::cout << static_cast<int>(*itPred) << " != " << static_cast<int>(*itReal) << std::endl;
    }
    return ::testing::AssertionSuccess();
}


TEST_F(TestCOMBI, testPrediction)
{
    const int lags = 5;
    const double validateSize = 1;

    //std::vector<double> dataValues{ 22, 44, 66, 88, 110, 132 }; //TODO: ONLY SCALED DATA WAAAT????
    std::vector<double> dataValues{ 140, 141, 142, 143, 144, 145, 146 };
    VectorXd data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
    auto timeSeries = convertToTimeSeries(data, lags);
    SplittedData splittedData = splitTimeSeries(timeSeries.first, timeSeries.second, validateSize);

    //std::cout << testGrowingLinearDependenceTS.getBestPolynomial() << std::endl;
    //std::cout << testGrowingLinearDependenceTS.predict(splittedData.xTest) << std::endl;
    //std::cout << splittedData.yTest << std::endl;
    auto res = testGrowingLinearDependenceTS.predict(splittedData.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, splittedData.yTest, 8)); 


    dataValues = { 59, 58, 57, 56, 55, 54, 53 };
    data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
    timeSeries = convertToTimeSeries(data, lags);
    splittedData = splitTimeSeries(timeSeries.first, timeSeries.second, validateSize);
    res = testDescendingLinearDependenceTS.predict(splittedData.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, splittedData.yTest, 0));

/*
    dataValues = { 100, 108, 116, 124, 132, 140, 148 };
    data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
    timeSeries = convertToTimeSeries(data, lags);
    splittedData = splitTimeSeries(timeSeries.first, timeSeries.second, validateSize);
    res = testDescendingLinearDependenceTS.predict(splittedData.xTest);
    res = testGrowingLinearDependenceWithNoiseTS.predict(splittedData.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, splittedData.yTest, 0));
*/

    
}


int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

// TODO: test threads


// TODO: test save/load

/*
auto res = combi.getBestPolynomial(); 
    combi.save("model1.txt");
    combi.load("model1.txt");
    auto res2 = combi.getBestPolynomial();

    ASSERT_STREQ(res.c_str(), res2.c_str());

*/



// TODO: test getBestPolynomial()