#include "test_setup.h"
#include <multi.h>

class TestMULTI : public TestGmdhModel {
protected:
    void setModel() override {
        if (SKIP_MULTI)
            GTEST_SKIP();
        testModel = new MULTI;
    }
public:
    static bool SKIP_MULTI;
};

bool TestMULTI::SKIP_MULTI = false;


TEST_F(TestMULTI, testFitOnDeath) { 
    auto testData = getTestData();
    GTEST_EXPECT_NO_DEATH({
        static_cast<MULTI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    }) << "[ TEST_MSG ]: death fit";
    if (HasFailure())
        TestMULTI::SKIP_MULTI = true;
}

TEST_F(TestMULTI, testPrediction) {
    auto testData = getTestData();
    static_cast<MULTI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testPredict(testData.dataValues.xTest, testData.realPredValues);
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestMULTI, testSave) {
    auto testData = getTestData();
    static_cast<MULTI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testSave();
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}

TEST_F(TestMULTI, testLoad) {
    auto testData = getTestData();
    static_cast<MULTI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    testModel->save("gtest_model.log");
    auto errorMsg = testLoad("gtest_model.log", "", "");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
    std::remove("gtest_model.log");
}

TEST_F(TestMULTI, testGetBestPolinomial) {
    auto testData = getTestData();
    static_cast<MULTI*>(testModel)->fit(testData.dataValues.xTrain, testData.dataValues.yTrain);
    auto errorMsg = testGetBestPolinomial("");
    EXPECT_TRUE(errorMsg.empty()) << errorMsg;
}