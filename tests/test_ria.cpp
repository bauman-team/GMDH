#include "test_setup.h"
#include <ria.h>

class TestRIA : public TestGmdhModel
{
protected:
    void setModel(std::vector<double> values) override {
        RIA *ria = new RIA;
        ria->fit(testModel.first.dataValues.xTrain, testModel.first.dataValues.yTrain);
        testModel.second = ria;
    }
};


TEST_F(TestRIA, testPrediction)
{
    VectorXd standartData = Map<VectorXd, Unaligned>(testModel.first.realPredValues.data(), testModel.first.realPredValues.size());
    auto res = testModel.second->predict(testModel.first.dataValues.xTest);
    EXPECT_TRUE(PredictionEvaluation(res, standartData, 3)); 
       
}
// TODO: test save(), load()