#include "test_setup.h"
#include <mia.h>
#include <ria.h>

class TestMIA : public TestGmdhModel {
protected:
    void setModel() override {
        if (SKIP_MIA)
            GTEST_SKIP();
        testModel = new MIA;
    }
public:
    static bool SKIP_MIA;
};

bool TestMIA::SKIP_MIA = false;

TEST_F(TestMIA, testFitOnDeath) { 
    auto criterion = GMDH::Criterion(CriterionType::regularity);
    auto testData = getTestDataFromFile();
    GTEST_EXPECT_NO_DEATH({
        static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion, 10, PolynomialType::linear_cov, 0.5, 1, 1, 1);
    }) << "[ TEST_MSG ]: death fit";
    if (HasFailure())
        TestMIA::SKIP_MIA = true;
    EXPECT_THROW(static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion, 2), std::invalid_argument) << "[ TEST_MSG ]: checking kBest value < 3";
    MatrixXd X{ {1, 2}, {4, 5} };
    EXPECT_THROW(static_cast<MIA*>(testModel)->fit(X, testData.dataValues.yTrain, criterion), std::invalid_argument) << "[ TEST_MSG ]: checking x.cols() >= 3";
}

TEST_F(TestMIA, testPrediction) {
    auto testData = getTestData();
    static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestMIA, testSave) {
    auto testData = getTestData();
    static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testSave();
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestMIA, testLoad) {
    auto testData = getTestData();
    static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    testModel->save("gtest_model.log");
    RIA ria;
    ria.save("gtest_diff_model.log");
    auto errorMsg = testLoad("gtest_model.log", "gtest_diff_model.log", "");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
    std::remove("gtest_model.log");
    std::remove("gtest_diff_model.log");
}

TEST_F(TestMIA, testGetBestPolinomial) {
    auto criterion = GMDH::Criterion(CriterionType::regularity);
    auto testData = getTestDataFromFile();
    static_cast<MIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain, criterion, 10);
    auto errorMsg = testGetBestPolinomial("");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}