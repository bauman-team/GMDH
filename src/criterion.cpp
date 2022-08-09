#include "criterion.h"

namespace GMDH {

RegularityCriterion::RegularityCriterion(double _test_size, bool _shuffle, int _random_seed) : RegularityCriterionTS(_test_size)
{
    shuffle = _shuffle;
    random_seed = _random_seed;
}

VectorXd Criterion::internalCriterion(const MatrixXd& x_train, const VectorXd& y_train) const
{ 
    return x_train.colPivHouseholderQr().solve(y_train); // TODO: add parameter to choose solve algorithm
}

std::pair<double, VectorXd> RegularityCriterion::calculate(const MatrixXd& x, const VectorXd& y) const
{
    return getCriterionValue(splitData(x, y, test_size, shuffle, random_seed));
}

std::pair<double, VectorXd> RegularityCriterionTS::getCriterionValue(const splitted_data& data) const
{
    VectorXd coeffs = internalCriterion(data.x_train, data.y_train);
    //vec coeffs(data.x_test.n_cols, fill::randn);
    VectorXd y_pred = data.x_test * coeffs;
    return std::pair<double, VectorXd>(((data.y_test - y_pred).array().square().sum() / data.y_test.array().square().sum()), coeffs);
}

RegularityCriterionTS::RegularityCriterionTS(double _test_size)
{
    if (_test_size > 0 && _test_size < 1)
        test_size = _test_size;
    else
        throw; // TODO: exception???
}

std::pair<double, VectorXd> RegularityCriterionTS::calculate(const MatrixXd& x, const VectorXd& y) const
{
    return getCriterionValue(splitTsData(x, y, test_size));
}
}