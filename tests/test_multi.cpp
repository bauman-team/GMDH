#include "test_setup.h"
#include <multi.h>

class TestMULTI : public TestGmdhModel
{
protected:
    void setModel(std::vector<double> values) override {
        MULTI *multi = new MULTI;
        multi->fit(testModel.first.dataValues.xTrain, testModel.first.dataValues.yTrain);
        testModel.second = multi;
    }
};


TEST_F(TestMULTI, testPrediction)
{
    VectorXd standartData = Map<VectorXd, Unaligned>(testModel.first.realPredValues.data(), testModel.first.realPredValues.size());
    auto res = testModel.second->predict(testModel.first.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 
       
}