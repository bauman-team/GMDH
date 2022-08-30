namespace GMDH {

enum class Solver { fast, accurate, balanced };

enum class CriterionType {regularity, symRegularity, stability, symStability, unbiasedOutputs, symUnbiasedOutputs,
                          unbiasedCoeffs, absoluteStability, symAbsoluteStability}; // TODO: maybe add cross validation criterion

struct BufferValues {
    VectorXd coeffsTrain;
    VectorXd coeffsTest;
    VectorXd coeffsAll;
    VectorXd yPredTrainByTrain;
    VectorXd yPredTrainByTest;
    VectorXd yPredTestByTrain;
    VectorXd yPredTestByTest;
};

class GMDH_API Criterion {
protected:

    CriterionType criterionType;
    Solver solver;

    VectorXd findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const;

    PairDVXd getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                       CriterionType _criterionType, BufferValues& bufferValues) const;

    PairDVXd regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                        BufferValues& bufferValues, bool inverseSplit = false) const;

    PairDVXd symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                           BufferValues& bufferValues) const;

    PairDVXd stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                       BufferValues& bufferValues, bool inverseSplit = false) const;

    PairDVXd symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                          BufferValues& bufferValues) const;

    PairDVXd unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                             BufferValues& bufferValues) const;

    PairDVXd symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                                BufferValues& bufferValues) const;

    PairDVXd unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                            BufferValues& bufferValues) const;

    PairDVXd absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                               BufferValues& bufferValues) const;

    PairDVXd symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                                  BufferValues& bufferValues) const;

public:
    Criterion() {};
    Criterion(CriterionType _criterionType, Solver _solver = Solver::balanced);

    std::string getClassName() const;

    virtual PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                               const VectorXd& yTrain, const VectorXd& yTest) const;
};


class GMDH_API ParallelCriterion : public Criterion {
    CriterionType secondCriterionType;
    double alpha;

public:
    ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, 
                        double _alpha = 0.5, Solver _solver = Solver::balanced);

    PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                       const VectorXd& yTrain, const VectorXd& yTest) const override;
};

class GMDH_API SequentialCriterion : public Criterion {
    CriterionType secondCriterionType;

public:
    SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver = Solver::balanced);

    PairDVXd recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                         const VectorXd& yTrain, const VectorXd& yTest, const VectorXd& _coeffsTrain) const;
};
}