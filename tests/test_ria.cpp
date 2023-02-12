#include "test_setup.h"
#include <ria.h>

class TestRIA : public TestGmdhModel {
protected:
    void setModel() override {
        if (SKIP_RIA)
            GTEST_SKIP();
        testModel = new RIA;
    }
public:
    static bool SKIP_RIA;
};

bool TestRIA::SKIP_RIA = false;

TEST_F(TestRIA, testFitOnDeath) { 
    auto testData = getTestDataFromFile();
    GTEST_EXPECT_NO_DEATH({
        static_cast<RIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    }) << "[ TEST_MSG ]: death fit";
    if (HasFailure())
        TestRIA::SKIP_RIA = true;
}

TEST_F(TestRIA, testPrediction) {
    auto testData = getTestData();
    static_cast<RIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestRIA, testSave) {
    auto testData = getTestData();
    static_cast<RIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testSave();
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestRIA, testLoad) {
    auto testData = getTestData();
    static_cast<RIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    testModel->save("gtest_model.log");
    auto errorMsg = testLoad("gtest_model.log", "", "");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
    std::remove("gtest_model.log");
}

TEST_F(TestRIA, testGetBestPolinomial) {
    auto testData = getTestDataFromFile();
    static_cast<RIA*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testGetBestPolinomial("");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}