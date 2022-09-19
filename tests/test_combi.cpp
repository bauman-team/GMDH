#include "test_setup.h"
#include <combi.h>

class TestCOMBI : public TestGmdhModel
{
protected:
    void setModel(std::vector<double> values) override {
        COMBI *combi = new COMBI;
        combi->fit(testModel.first.dataValues.xTrain, testModel.first.dataValues.yTrain);
        testModel.second = combi;
    }
};


TEST_F(TestCOMBI, testPrediction)
{
    VectorXd standartData = Map<VectorXd, Unaligned>(testModel.first.realPredValues.data(), testModel.first.realPredValues.size());
    auto res = testModel.second->predict(testModel.first.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 
       
}

// TODO: test save(), load()