#pragma once
#include <gtest/gtest.h>
#include <gmdh.h>
#include <iostream>

using namespace GMDH;
using namespace Eigen;


class TestGmdhModel : public ::testing::Test
{
protected:
    struct TestData {
        SplittedData dataValues; 
        int lags;
        Solver solverFunc;
        std::vector<double> realPredValues;
        Criterion criterion;
    };
	void SetUp()
	{
        TestData data;

        /* SET MODEL FOR PREDICT GROWING LINEAR DEPENDENCE TIME SERIES */
        data.lags = 5;
        data.solverFunc = Solver::fast;
        data.realPredValues = { 16, 17 };
        std::vector<double> values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };

        VectorXd valuesVec = Map<VectorXd, Unaligned>(values.data(), values.size());
        auto timeSeries = timeSeriesTransformation(valuesVec, data.lags);
        SplittedData splittedData = splitData(timeSeries.first, timeSeries.second);
        data.dataValues = splittedData;
        data.criterion = Criterion(CriterionType::regularity);
        testModel.first = data;
        setModel(values);

	}
	void TearDown()
	{
		// if it need clear memory
        delete testModel.second;
	}
    virtual void setModel(std::vector<double> values) = 0;
	std::pair<TestData, GmdhModel*> testModel;
};


::testing::AssertionResult PredictionEvaluation(VectorXd predict, VectorXd real, int precision);