#pragma once
#include <gtest/gtest.h>
#include <gmdh.h>
#include <iostream>

#define GTEST_EXPECT_NO_DEATH(statement) \
    EXPECT_EXIT({{ statement } ::exit(EXIT_SUCCESS); }, ::testing::ExitedWithCode(0), "")

using namespace GMDH;
using namespace Eigen;

extern const CriterionType allCriterionTypes[9];

class TestGmdhModel : public ::testing::Test
{
protected:
    struct TestData {
        SplittedData dataValues; 
        VectorXd realPredValues;
    };
    GmdhModel* testModel;

	void SetUp()
	{
        if (SKIP_FIXTURES)
            GTEST_SKIP();

        setModel();
	}
	void TearDown()
	{
		// if it need clear memory
        delete testModel;
	}

    std::string testSave();
    
    std::string testLoad(std::string modelPath, std::string differentModelPath, std::string bestPolinomial);
    
    std::string testPredict(MatrixXd test, VectorXd predValues);
    
    std::string testLongTermPredict(MatrixXd test, VectorXd predValues);
    
    std::string testGetBestPolinomial(std::string bestPolinomial);

    virtual void setModel() = 0;

    TestData getTestData();
    
    TestData getTestDataFromFile();
public:
    static bool SKIP_FIXTURES;
}; // TODO: solution matrix on tests result

::testing::AssertionResult PredictionEvaluation(VectorXd predict, VectorXd real, int precision);