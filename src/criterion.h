#include "gmdh.h"

namespace GMDH {
class Criterion {

protected:
    VectorXd internalCriterion(const MatrixXd& x_train, const VectorXd& y_train) const;

public:
    virtual std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const = 0;
};

class RegularityCriterionTS : public Criterion
{
protected:
    double test_size;

    std::pair<double, VectorXd> getCriterionValue(const splitted_data& data) const;
public:
    RegularityCriterionTS(double _test_size = 0.33);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};

class RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int random_seed;

public:
    RegularityCriterion(double _test_size = 0.33, bool _shuffle = true, int _random_seed = 0);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};
}