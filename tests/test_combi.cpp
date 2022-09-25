#include "test_setup.h"
#include <combi.h>
#include <ria.h>

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

TEST_F(TestCOMBI, testSaveWithErrorFilePath)
{
    EXPECT_EQ(testModel.second->save("."), 1);
}

TEST_F(TestCOMBI, testSaveExistFile)
{
    EXPECT_EQ(testModel.second->save("combi_test.txt"), 0);
    std::remove("combi_test.txt");
}

TEST_F(TestCOMBI, testLoadWithErrorFilePath)
{
    EXPECT_EQ(testModel.second->load("."), 1);
    EXPECT_EQ(testModel.second->load("multi.txt"), 1);
}

TEST_F(TestCOMBI, testLoadWithCorruptedFile)
{
    std::ofstream file("combi_error_test.txt");
    file << "test\n";
    file.close();
    EXPECT_EQ(testModel.second->load("combi_error_test.txt"), 2);
    std::remove("combi_error_test.txt");
}

TEST_F(TestCOMBI, testLoadWithDiffModelFile)
{
    RIA ria;
    ria.save("ria.txt");
    EXPECT_EQ(testModel.second->load("ria.txt"), 3);
    std::remove("ria.txt");
}

TEST_F(TestCOMBI, testLoadExistFile)
{
    testModel.second->save("combi_test.txt");
    EXPECT_EQ(testModel.second->load("combi_test.txt"), 0);
    std::remove("combi_test.txt");
}