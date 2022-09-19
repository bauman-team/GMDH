namespace GMDH {

class GmdhModel;

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

    virtual VectorC getBestCombinations(VectorC& combinations, const SplittedData& data, const std::function<MatrixXd(const MatrixXd&, const VectorU16&)> func, int k) const;

    virtual PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                               const VectorXd& yTrain, const VectorXd& yTest) const;

public:
    Criterion() {};
    Criterion(CriterionType _criterionType, Solver _solver = Solver::balanced);

    friend class GmdhModel;
};


class GMDH_API ParallelCriterion : public Criterion {
    CriterionType secondCriterionType;
    double alpha;

    PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                       const VectorXd& yTrain, const VectorXd& yTest) const override;
public:
    ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, 
                        double _alpha = 0.5, Solver _solver = Solver::balanced);
};

class GMDH_API SequentialCriterion : public Criterion {
    CriterionType secondCriterionType;

    PairDVXd recalculate(const MatrixXd& xTrain, const MatrixXd& xTest,
        const VectorXd& yTrain, const VectorXd& yTest, const VectorXd& _coeffsTrain) const;
    
    VectorC getBestCombinations(VectorC& combinations, const SplittedData& data, const std::function<MatrixXd(const MatrixXd&, const VectorU16&)> func, int k) const override;
public:
    SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver = Solver::balanced);
};
}