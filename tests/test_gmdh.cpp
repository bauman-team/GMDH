#include "test_setup.h"

bool TestGmdhModel::SKIP_FIXTURES = false;

TEST(testTimeSeriesTransformation, testCorrectData) {
    std::pair<VectorXd, int> origData = { VectorXd{{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }}, 4 };
    GTEST_EXPECT_NO_DEATH({
        auto timeSeries = timeSeriesTransformation(origData.first, origData.second);
        
        for (auto i = 0; i != origData.first.size() - origData.second; ++i) {
            for (auto j = 0; j != origData.second; ++j)
                ASSERT_EQ(origData.first[j + i], timeSeries.first.row(i)[j]);
            ASSERT_EQ(origData.first[i + origData.second], timeSeries.second[i]);
        }
    }) << "[ TEST_MSG ]: death timeSeriesTransformation";
    if (HasFailure())
        TestGmdhModel::SKIP_FIXTURES = true;
}

TEST(testTimeSeriesTransformation, testIncorrectData) {
    std::pair<VectorXd, int> origData = { VectorXd{{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }}, 4 };
    VectorXd errorData;
    EXPECT_THROW(timeSeriesTransformation(errorData, 1), std::invalid_argument) << "[ TEST_MSG ]: empty timeSeries VectorXd in argument"; 
    EXPECT_THROW(timeSeriesTransformation(origData.first, -1), std::invalid_argument) << "[ TEST_MSG ]: negative number of lags";
    EXPECT_THROW(timeSeriesTransformation(origData.first, 0), std::invalid_argument) << "[ TEST_MSG ]: zero number of lags";
    EXPECT_THROW(timeSeriesTransformation(origData.first, origData.first.size()), std::invalid_argument) << "[ TEST_MSG ]: number of lags == size of timeSeries"; 
    EXPECT_THROW(timeSeriesTransformation(origData.first, origData.first.size() + 1), std::invalid_argument) << "[ TEST_MSG ]: lags > timeSeries size"; 
}

TEST(testSplitData, testCorrectData) {
    MatrixXd X{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    VectorXd y{{10, 11, 12}};
    GTEST_EXPECT_NO_DEATH({
        auto splited_data = splitData(X, y, 0.33);

        ASSERT_TRUE(splited_data.xTrain == X({ 0, 1 }, Eigen::all));
        ASSERT_TRUE(splited_data.xTest == X({ 2 }, Eigen::all));
        ASSERT_TRUE(splited_data.yTrain == y({ 0, 1 }));
        ASSERT_TRUE(splited_data.yTest == y({ 2 }));
    }) << "[ TEST_MSG ]: death splitData";
    if (HasFailure())
        TestGmdhModel::SKIP_FIXTURES = true;
}

TEST(testSplitDataWithShuffle, testCorrectData) {
    MatrixXd X{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    VectorXd y{{10, 11, 12}};
    GTEST_EXPECT_NO_DEATH({
        auto splited_data = splitData(X, y, 0.66, true, 1); 
        auto splited_data2 = splitData(X, y, 0.66, true, 1);

        ASSERT_TRUE(splited_data.xTrain == splited_data2.xTrain);
        ASSERT_TRUE(splited_data.xTest == splited_data2.xTest);
        ASSERT_TRUE(splited_data.yTrain == splited_data2.yTrain);
        ASSERT_TRUE(splited_data.yTest == splited_data2.yTest);
    }) << "[ TEST_MSG ]: death splitData with shuffle";
}


TEST(testSplitData, testIncorrectData) {
    MatrixXd X{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    VectorXd y{{10, 11, 12}};
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), 0.1), std::invalid_argument) << "[ TEST_MSG ]: empty test arrays because of small test_size"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), 0.9), std::invalid_argument) << "[ TEST_MSG ]: empty train arrays because of large test_size"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0 }), 0.5), std::invalid_argument) << "[ TEST_MSG ]: different X rows and y size values"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), 0), std::invalid_argument) << "[ TEST_MSG ]: test_size = 0"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), -1), std::invalid_argument) << "[ TEST_MSG ]: test_size < 0"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), 1), std::invalid_argument) << "[ TEST_MSG ]: test_size = 1"; 
    EXPECT_THROW(splitData(X({ 0, 1 }, Eigen::all), X({ 2 }, Eigen::all)({ 0, 1 }), 10), std::invalid_argument) << "[ TEST_MSG ]: test_size > 1"; 
}

TEST(testValidateInputData, testCorrectData) {
    double testSize = 0.5, limit = 0.3;
    int pAverage = 1, threads = 2, verbose = 0, kBest = 4;
    EXPECT_EQ(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), 0);
    threads = 0, verbose = 1;
    EXPECT_EQ(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), 0);
    threads = 100, verbose = 6;
    EXPECT_EQ(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), 0);
    threads = -1, verbose = -1;
    EXPECT_EQ(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), 0);
}

TEST(testValidateInputData, testIncorrectData) {
    double testSize = 0.5, limit = 0.3;
    int pAverage = 1, threads = 2, verbose = 0, kBest = 4;

    testSize = 0;
    EXPECT_THROW(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), std::invalid_argument);
    testSize = 1;
    EXPECT_THROW(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), std::invalid_argument);
    testSize = 0.5;

    pAverage = 0;
    EXPECT_THROW(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), std::invalid_argument);
    pAverage = 1;

    limit = -1;
    EXPECT_THROW(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), std::invalid_argument);
    limit = 0;

    kBest = 0;
    EXPECT_THROW(validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest), std::invalid_argument);
}

TEST(testParallelCriterion, testConstructorIncorrectData) {
    EXPECT_THROW(ParallelCriterion criterion(CriterionType::regularity, CriterionType::stability, -1), std::invalid_argument) << "[ TEST_MSG ]: alpha < 0"; 
    EXPECT_THROW(ParallelCriterion criterion(CriterionType::regularity, CriterionType::stability, 0), std::invalid_argument) << "[ TEST_MSG ]: alpha == 0"; 
    EXPECT_THROW(ParallelCriterion criterion(CriterionType::regularity, CriterionType::stability, 1), std::invalid_argument) << "[ TEST_MSG ]: alpha == 1"; 
    EXPECT_THROW(ParallelCriterion criterion(CriterionType::regularity, CriterionType::stability, 10), std::invalid_argument) << "[ TEST_MSG ]: alpha > 1"; 
}

