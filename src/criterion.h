namespace GMDH {

enum Solver { fast, accurate, balanced };

class GMDH_API Criterion {

protected:
    Solver solver;
    VectorXd findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const;

public:
    virtual PairDVXd calculate(const MatrixXd& x, const VectorXd& y) const = 0;
};

class GMDH_API RegularityCriterionTS : public Criterion
{
protected:
    double testSize;

    PairDVXd getCriterionValue(const SplittedData& data) const;
public:
    RegularityCriterionTS(double _testSize = 0.33, Solver _solver = Solver::balanced);
    PairDVXd calculate(const MatrixXd& x, const VectorXd& y) const override;
};

class GMDH_API RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int randomSeed;

public:
    RegularityCriterion(double _testSize = 0.33, Solver _solver = Solver::balanced, bool _shuffle = true, int _randomSeed = 0);
    PairDVXd calculate(const MatrixXd& x, const VectorXd& y) const override;
};
}