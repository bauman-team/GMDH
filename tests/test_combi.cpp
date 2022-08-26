#include <gtest/gtest.h>
#include <combi.h>
#include <iostream>

using namespace GMDH;
using namespace Eigen;


class TestCOMBI : public ::testing::Test
{
protected:
    struct TestData {
        SplittedData dataValues; 
        int lags;
        double testSize, validateSize;
        Solver solverFunc;
        std::vector<double> realPredValues;
    };
	void SetUp()
	{
        TestData data;

        /* SET MODEL FOR PREDICT GROWING LINEAR DEPENDENCE TIME SERIES */
        data.lags = 5; data.testSize = 0.33; data.validateSize = 0.2;
        data.solverFunc = Solver::fast;
        data.realPredValues = { 16, 17 };
        std::vector<double> values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
        testModels.push_back(getModelTS(data, values));


        /* SET MODEL FOR PREDICT DESCENDING LINEAR DEPENDENCE TIME SERIES */
        data.realPredValues = { 3, 2, 1 };
        values = { 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
        testModels.push_back(getModelTS(data, values));


        /* SET MODEL FOR PREDICT GROWING LINEAR DEPENDENCE WITH NOISE TIME SERIES */
        data.realPredValues = { 141.472, 146.277 };
        values = { 10, 19, 31, 40, 50, 62, 71, 79, 89, 98, 106, 120, 124, 129, 140, 146 };
        testModels.push_back(getModelTS(data, values));


        /* SET MODEL FOR PREDICT LINEAR POLINOMIAL DEPENDENCE */


        /* SET MODEL FOR PREDICT LINEAR POLINOMIAL DEPENDENCE WITH NOISE */


        //data = Map<VectorXd, Unaligned>(dataValues.data(), dataValues.size());
        
	}
	void TearDown()
	{
		// if it need clear memory
	}
    std::pair<COMBI, TestData> getModelTS(TestData _data, std::vector<double> values) {
        VectorXd data = Map<VectorXd, Unaligned>(values.data(), values.size());
        auto timeSeries = convertToTimeSeries(data, _data.lags);
        SplittedData splittedData = splitData(timeSeries.first, timeSeries.second, _data.validateSize);
        COMBI combi;
        auto criterion = Criterion(CriterionType::regularity);
        combi.fit(splittedData.xTrain, splittedData.yTrain, criterion, _data.testSize, 0, 0, 2, 1);
        _data.dataValues = splittedData;
        return {combi, _data};
    }
    void setModel() { // TODO: for polinomial models

    }
	std::vector<std::pair<COMBI, TestData>> testModels; // TODO: add polinomial tests
};


::testing::AssertionResult PredictionEvaluation(VectorXd predict, VectorXd real, int precision) // TODO: check precision ????
{
    auto truncLast{ 10. };
    truncLast = std::pow(truncLast, precision);
    for (auto itPred = predict.begin(), itReal = real.begin(); itReal != real.end(); ++itPred, ++itReal) {
        //std::cout << *itPred << '\n' << *itReal << std::endl;
        if (static_cast<int64_t>(round(*itPred * truncLast)) != static_cast<int64_t>(round(*itReal * truncLast)))
		    return ::testing::AssertionFailure();
            //std::cout << static_cast<int>(*itPred) << " != " << static_cast<int>(*itReal) << std::endl;
    }
    return ::testing::AssertionSuccess();
}


TEST_F(TestCOMBI, testPrediction)
{
    //std::cout << testGrowingLinearDependenceTS.getBestPolynomial() << std::endl;
    //std::cout << testGrowingLinearDependenceTS.predict(splittedData.xTest) << std::endl;
    //std::cout << splittedData.yTest << std::endl;
    VectorXd standartData = Map<VectorXd, Unaligned>(testModels[0].second.realPredValues.data(), testModels[0].second.realPredValues.size());
    auto res = testModels[0].first.predict(testModels[0].second.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 


    standartData = Map<VectorXd, Unaligned>(testModels[1].second.realPredValues.data(), testModels[1].second.realPredValues.size());
    res = testModels[1].first.predict(testModels[1].second.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 
    

    standartData = Map<VectorXd, Unaligned>(testModels[2].second.realPredValues.data(), testModels[2].second.realPredValues.size());
    res = testModels[2].first.predict(testModels[2].second.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 
    
    //std::cout << testGrowingLinearDependenceTS.getBestPolynomial() << std::endl;
    //std::cout << testGrowingLinearDependenceTS.predict(splittedData.xTest) << std::endl;
    //std::cout << splittedData.yTest << std::endl;


    
}


int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

// TODO: test splitData

// TODO: test fit

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