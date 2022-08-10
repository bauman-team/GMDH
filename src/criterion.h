#include "gmdh.h"

namespace GMDH {

enum Solver { fast, accurate, balanced };

class Criterion {

protected:
    Solver solver;
    VectorXd findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const;

public:
    virtual std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const = 0;
};

class RegularityCriterionTS : public Criterion
{
protected:
    double testSize;

    std::pair<double, VectorXd> getCriterionValue(const SplittedData& data) const;
public:
    RegularityCriterionTS(double _testSize = 0.33, Solver _solver = Solver::balanced);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};

class RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int randomSeed;

public:
    RegularityCriterion(double _testSize = 0.33, Solver _solver = Solver::balanced, bool _shuffle = true, int _randomSeed = 0);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};
}