#include "test_setup.h"

const CriterionType allCriterionTypes[9] = {CriterionType::regularity, CriterionType::symRegularity, CriterionType::stability, CriterionType::symStability, 
                                            CriterionType::unbiasedOutputs, CriterionType::symUnbiasedOutputs, CriterionType::unbiasedCoeffs, 
                                            CriterionType::absoluteNoiseImmunity, CriterionType::symAbsoluteNoiseImmunity};

std::string TestGmdhModel::testSave() {
    GTEST_EXPECT_NO_DEATH({
        testModel->save("gtest_model.log");
        std::remove("gtest_model.log");
    });
    if (HasFailure())
        return "[ TEST_MSG ]: death save";
    EXPECT_EQ(testModel->save("gtest_model.log"), 0);
    std::remove("gtest_model.log");
    if (HasFailure())
        return "[ TEST_MSG ]: save return error code: expect 0";
    GTEST_EXPECT_NO_DEATH({
        if (testModel->save(".") != 1)
            exit(1);
    });
    if (HasFailure())
        return "[ TEST_MSG ]: save return error code: expect 1";
    return "";
}

std::string TestGmdhModel::testLoad(std::string modelPath, std::string differentModelPath, std::string bestPolinomial) {
    GTEST_EXPECT_NO_DEATH({
        testModel->load(modelPath);
    });
    if (HasFailure())
        return "[ TEST_MSG ]: death load";

    // TODO: test bestPolinom 
    
    EXPECT_EQ(testModel->load(modelPath), 0);
    if (HasFailure())
        return "[ TEST_MSG ]: load return error code: expect 0";
    GTEST_EXPECT_NO_DEATH({
        if (testModel->load(".") != 1 || testModel->load("multi.txt") != 1)
            exit(1);
    });
    if (HasFailure())
        return "[ TEST_MSG ]: load return error code: expect 1";
    std::ofstream file("combi_error_test.txt");
    file << "test\n";
    file.close();
    GTEST_EXPECT_NO_DEATH({
        if (testModel->load("combi_error_test.txt") != 2)
            exit(1);
    });
    std::remove("combi_error_test.txt");
    if (HasFailure())
        return "[ TEST_MSG ]: load return error code: expect 2";
    if (differentModelPath != "") {
        GTEST_EXPECT_NO_DEATH({
            if (testModel->load(differentModelPath) != 3)
                exit(1);
        });
        if (HasFailure())
            return "[ TEST_MSG ]: load return error code: expect 3";
    }
    return "";
}

std::string TestGmdhModel::testPredict(MatrixXd test, VectorXd predValues) {
    GTEST_EXPECT_NO_DEATH({
        testModel->predict(test);
    });
    if (HasFailure())
        return "[ TEST_MSG ]: death predict";
    auto res = testModel->predict(test);
    EXPECT_TRUE(PredictionEvaluation(res, predValues, 3));
    if (HasFailure())
        return "[ TEST_MSG ]: predict erroneous data";
    return "";
}

std::string TestGmdhModel::testLongTermPredict(MatrixXd test, VectorXd predValues) {
    GTEST_EXPECT_NO_DEATH({
        testModel->predict(test.row(0).transpose(), 2);
    });
    if (HasFailure())
        return "[ TEST_MSG ]: death predict with lags";
    auto res = testModel->predict(test.row(0).transpose(), 2);
    EXPECT_TRUE(PredictionEvaluation(res, predValues, 3));
    if (HasFailure())
        return "[ TEST_MSG ]: predict with lags erroneous data";
    EXPECT_THROW(testModel->predict(test.row(0).transpose(), 0), std::invalid_argument);
    EXPECT_THROW(testModel->predict(test.row(0).transpose(), -1), std::invalid_argument);
    if (HasFailure())
        return "[ TEST_MSG ]: predict can't handle nums <= 0";
    return "";
}

std::string TestGmdhModel::testGetBestPolinomial(std::string bestPolinomial) { // TODO: test with compare example
    GTEST_EXPECT_NO_DEATH({
        testModel->getBestPolynomial();
    });
    if (HasFailure())
        return "[ TEST_MSG ]: death getBestPolynomial";
    return "";
}

TestGmdhModel::TestData TestGmdhModel::getTestData() {
    TestData data;
    data.realPredValues = VectorXd{{ 16, 17 }};
    VectorXd valuesVec{{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }};
    auto lags = 5;
    auto timeSeries = timeSeriesTransformation(valuesVec, lags);
    SplittedData splittedData = splitData(timeSeries.first, timeSeries.second);
    data.dataValues = splittedData;
    return data;
}

TestGmdhModel::TestData TestGmdhModel::getTestDataFromFile() {
    TestData testData;
    std::ifstream dataStream;
    
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
    }
    int lags = 8;
    double validateSize = 0.2;
    double testSize = 0.33;
    auto timeSeries = GMDH::timeSeriesTransformation(data, lags);
    SplittedData splittedData = GMDH::splitData(timeSeries.first, timeSeries.second, validateSize);

    testData.dataValues = splittedData;
    return testData;
}

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

int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}