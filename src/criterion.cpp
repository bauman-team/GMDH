#include "criterion.h"

namespace GMDH {

RegularityCriterion::RegularityCriterion(double _testSize, Solver _solver, bool _shuffle, int _randomSeed) : RegularityCriterionTS(_testSize)
{
    solver = _solver;
    shuffle = _shuffle;
    randomSeed = _randomSeed;
}

VectorXd Criterion::findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const
{ 
    VectorXd coeffs;
    switch (solver) {
    case Solver::accurate:
        coeffs = xTrain.fullPivHouseholderQr().solve(yTrain);
        break;
    case Solver::balanced:
        coeffs = xTrain.colPivHouseholderQr().solve(yTrain);
        break;
    case Solver::fast:
        coeffs =  xTrain.householderQr().solve(yTrain);
        break;
    }
    return coeffs;
}

std::pair<double, VectorXd> RegularityCriterion::calculate(const MatrixXd& x, const VectorXd& y) const
{
    return getCriterionValue(splitData(x, y, testSize, shuffle, randomSeed));
}

std::pair<double, VectorXd> RegularityCriterionTS::getCriterionValue(const SplittedData& data) const
{
    VectorXd coeffs = findBestCoeffs(data.xTrain, data.yTrain);
    VectorXd yPred = data.xTest * coeffs;
    return std::pair<double, VectorXd>(((data.yTest - yPred).array().square().sum() / data.yTest.array().square().sum()), coeffs);
}

RegularityCriterionTS::RegularityCriterionTS(double _testSize, Solver _solver)
{
    solver = _solver;
    if (_testSize > 0 && _testSize < 1)
        testSize = _testSize;
    else // TODO: add warning about incorrect test_size value
        testSize = 0.33; 
}

std::pair<double, VectorXd> RegularityCriterionTS::calculate(const MatrixXd& x, const VectorXd& y) const
{
    return getCriterionValue(splitTimeSeries(x, y, testSize));
}
}