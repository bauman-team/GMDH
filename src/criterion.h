namespace GMDH {

enum class Solver { fast, accurate, balanced };

enum class CriterionType {regularity, symRegularity, stability, symStability, unbiasedOutputs, symUnbiasedOutputs,
                    unbiasedCoeffs, absoluteStability, symAbsoluteStability}; // TODO: maybe add cross validation criterion


struct TempValues {
    VectorXd coeffsTrain;
    VectorXd coeffsTest;
    VectorXd coeffsAll;
    VectorXd yPredTrainByTrain;
    VectorXd yPredTrainByTest;
    VectorXd yPredTestByTrain;
    VectorXd yPredTestByTest;
};

class Criterion {    
protected:

    CriterionType criterionType;
    Solver solver;

    VectorXd findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const;
    PairDVXd getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, CriterionType _criterionType, TempValues& tempValues) const;

    PairDVXd regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues, bool inverseSplit = false) const;
    PairDVXd symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues, bool inverseSplit = false) const;
    PairDVXd symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;
    PairDVXd symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const;

public:
    Criterion() {};
    Criterion(CriterionType _criterionType, Solver _solver = Solver::balanced);

    std::string getClassName() const;

    virtual PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const;
};


class ParallelCriterion : public Criterion {
    CriterionType secondCriterionType;
    double alpha;

public:
    ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, 
                        double _alpha = 0.5, Solver _solver = Solver::balanced);

    PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const override;
};

class SequentialCriterion : public Criterion {
    CriterionType secondCriterionType;

public:
    SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver = Solver::balanced);

    PairDVXd recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, const VectorXd& _coeffsTrain) const;
};
}