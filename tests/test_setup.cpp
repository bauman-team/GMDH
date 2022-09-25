#include "test_setup.h"

::testing::AssertionResult PredictionEvaluation(VectorXd predict, VectorXd real, int precision) // TODO: check precision ????
{
    auto truncLast{ 10. };
    truncLast = std::pow(truncLast, precision);
    for (auto itPred = predict.begin(), itReal = real.begin(); itReal != real.end(); ++itPred, ++itReal) {
        //std::cout << *itPred << '\n' << *itReal << std::endl;
        if (static_cast<int64_t>(round(*itPred * truncLast)) != static_cast<int64_t>(round(*itReal * truncLast)))
		    return ::testing::AssertionFailure();
            //std::cout << static_cast<int>(*itPred) << " != " << static_cast<int>(*itReal) << std::endl;
    }
    return ::testing::AssertionSuccess();
}

int main(int argc, char *argv[])
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}