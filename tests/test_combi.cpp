#include "test_setup.h"
#include <combi.h>
#include <ria.h>

class TestCOMBI : public TestGmdhModel {
protected:
    void setModel() override {
        testModel = new COMBI;
        if (SKIP_COMBI)
            GTEST_SKIP();
    }
public:
    static bool SKIP_COMBI;
};

bool TestCOMBI::SKIP_COMBI = false;


TEST_F(TestCOMBI, testFitOnDeath) { 
    auto testData = getTestData();
    GTEST_EXPECT_NO_DEATH({
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    }) << "[ TEST_MSG ]: death fit";
    if (HasFailure())
        TestCOMBI::SKIP_COMBI = true;
}

TEST_F(TestCOMBI, testPrediction) {
    auto testData = getTestData();
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestCOMBI, testLongTermPredict) {
    auto testData = getTestData();
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testLongTermPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestCOMBI, testPredictionError) {
    auto testData = getTestData();
    MatrixXd errorX;
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    EXPECT_THROW(testModel->predict(errorX), std::invalid_argument);
}

TEST_F(TestCOMBI, testSave) {
    auto testData = getTestData();
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testSave();
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestCOMBI, testLoad) {
    auto testData = getTestData();
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    testModel->save("gtest_model.log");
    RIA ria;
    ria.save("gtest_diff_model.log");
    auto errorMsg = testLoad("gtest_model.log", "gtest_diff_model.log", "");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
    std::remove("gtest_model.log");
    std::remove("gtest_diff_model.log");
}

TEST_F(TestCOMBI, testAllCriterionTypes) {
    
    auto testData = getTestData();
    for (auto i: allCriterionTypes) {
        auto criterion = GMDH::Criterion(i);
        GTEST_EXPECT_NO_DEATH({
            static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion);
        }) << "[ TEST_MSG ]: death fit with criterion #"+std::to_string(static_cast<int>(i));
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion);
        auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
        EXPECT_TRUE(errorMsg.empty()) << errorMsg+" with criterion #"+std::to_string(static_cast<int>(i));
    }
}

TEST_F(TestCOMBI, testParallelCriterionAllSolvers) {
    auto testData = getTestData();
    ParallelCriterion criterion1(CriterionType::regularity, CriterionType::unbiasedOutputs),
    criterion2(CriterionType::regularity, CriterionType::unbiasedOutputs, 0.5, Solver::accurate),
    criterion3(CriterionType::regularity, CriterionType::unbiasedOutputs, 0.5, Solver::fast);
    GTEST_EXPECT_NO_DEATH({
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion1);
    }) << "[ TEST_MSG ]: death fit with ParallelCriterion and Solver::balanced";
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion1);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg+" with ParallelCriterion and Solver::balanced";

    GTEST_EXPECT_NO_DEATH({
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion2);
    }) << "[ TEST_MSG ]: death fit with ParallelCriterion and Solver::accurate";
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion2);
    errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg+" with ParallelCriterion and Solver::accurate";

    GTEST_EXPECT_NO_DEATH({
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion3);
    }) << "[ TEST_MSG ]: death fit with ParallelCriterion and Solver::fast";
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion3);
    errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg+" with ParallelCriterion and Solver::fast";
}


TEST_F(TestCOMBI, testSequentialCriterion) {
    auto testData = getTestData();
    SequentialCriterion criterion(CriterionType::regularity, CriterionType::unbiasedOutputs);
    GTEST_EXPECT_NO_DEATH({
        static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion);
    }) << "[ TEST_MSG ]: death fit with SequentialCriterion";
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg+" with SequentialCriterion";
}


TEST_F(TestCOMBI, testGetBestPolinomial) {
    auto testData = getTestData();
    static_cast<COMBI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testGetBestPolinomial("");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}
