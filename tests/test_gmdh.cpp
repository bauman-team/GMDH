#include "test_setup.h"
#include <gmdh.h>

TEST(testTimeSeriesTransformation, testCorrectData)
{
    std::vector<double> values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::pair<VectorXd, int> origData = { Map<VectorXd, Unaligned>(values.data(), values.size()), 4 };
    auto timeSeries = timeSeriesTransformation(origData.first, origData.second);
    for (auto i = 0; i != origData.first.size() - origData.second; ++i) {
        for (auto j = 0; j != origData.second; ++j)
            EXPECT_EQ(origData.first[j + i], timeSeries.first.row(i)[j]); 
        EXPECT_EQ(origData.first[i + origData.second], timeSeries.second[i]);
    }
}

TEST(testTimeSeriesTransformation, testIncorrectData)
{
    std::vector<double> values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::pair<VectorXd, int> origData = { Map<VectorXd, Unaligned>(values.data(), values.size()), 4 };
    VectorXd errorData;
    EXPECT_THROW(timeSeriesTransformation(errorData, 1), std::invalid_argument);
    EXPECT_THROW(timeSeriesTransformation(origData.first, -1), std::invalid_argument);
    EXPECT_THROW(timeSeriesTransformation(origData.first, 0), std::invalid_argument);
    EXPECT_THROW(timeSeriesTransformation(origData.first, origData.first.size()), std::invalid_argument);
    EXPECT_THROW(timeSeriesTransformation(origData.first, origData.first.size() + 1), std::invalid_argument);
}