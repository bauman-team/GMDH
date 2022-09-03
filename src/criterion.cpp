#include "gmdh.h"

namespace GMDH {

    VectorXd Criterion::findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const {
        if ((solver == Solver::accurate))
            return xTrain.fullPivHouseholderQr().solve(yTrain);
        else if ((solver == Solver::balanced))
            return xTrain.colPivHouseholderQr().solve(yTrain);
        else
            return xTrain.householderQr().solve(yTrain);
    }

    PairDVXd Criterion::regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, 
                                   const VectorXd& yTest, BufferValues& bufferValues, bool inverseSplit) const {
        if (!inverseSplit) {
            if (bufferValues.coeffsTrain.size() == 0)
                bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (bufferValues.yPredTestByTrain.size() == 0)
                bufferValues.yPredTestByTrain = xTest * bufferValues.coeffsTrain;
            return PairDVXd((yTest - bufferValues.yPredTestByTrain).array().square().sum(), bufferValues.coeffsTrain);
        }
        else {
            if (bufferValues.coeffsTest.size() == 0)
                bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
            if (bufferValues.yPredTrainByTest.size() == 0)
                bufferValues.yPredTrainByTest = xTrain * bufferValues.coeffsTest;
            return PairDVXd((yTrain - bufferValues.yPredTrainByTest).array().square().sum(), bufferValues.coeffsTest);
        }
    }

    PairDVXd Criterion::symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                      const VectorXd& yTest, BufferValues& bufferValues) const {
        PairDVXd part1 = regularity(xTrain, xTest, yTrain, yTest, bufferValues);
        PairDVXd part2 = regularity(xTrain, xTest, yTrain, yTest, bufferValues, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                  const VectorXd& yTest, BufferValues& bufferValues, bool inverseSplit) const {
        if (!inverseSplit) {
            if (bufferValues.coeffsTrain.size() == 0)
                bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (bufferValues.yPredTrainByTrain.size() == 0)
                bufferValues.yPredTrainByTrain = xTrain * bufferValues.coeffsTrain;
            if (bufferValues.yPredTestByTrain.size() == 0)
                bufferValues.yPredTestByTrain = xTest * bufferValues.coeffsTrain;
            return PairDVXd((yTrain - bufferValues.yPredTrainByTrain).array().square().sum() +
                            (yTest - bufferValues.yPredTestByTrain).array().square().sum(), bufferValues.coeffsTrain);
        }
        else {
            if (bufferValues.coeffsTest.size() == 0)
                bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
            if (bufferValues.yPredTrainByTest.size() == 0)
                bufferValues.yPredTrainByTest = xTrain * bufferValues.coeffsTest;
            if (bufferValues.yPredTestByTest.size() == 0)
                bufferValues.yPredTestByTest = xTest * bufferValues.coeffsTest;
            return PairDVXd((yTrain - bufferValues.yPredTrainByTest).array().square().sum() +
                            (yTest - bufferValues.yPredTestByTest).array().square().sum(), bufferValues.coeffsTest);
        }
    }

    PairDVXd Criterion::symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                     const VectorXd& yTest, BufferValues& bufferValues) const {
        PairDVXd part1 = stability(xTrain, xTest, yTrain, yTest, bufferValues);
        PairDVXd part2 = stability(xTrain, xTest, yTrain, yTest, bufferValues, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                        const VectorXd& yTest, BufferValues& bufferValues) const {
        if (bufferValues.coeffsTrain.size() == 0)
            bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (bufferValues.coeffsTest.size() == 0)
            bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (bufferValues.yPredTestByTrain.size() == 0)
            bufferValues.yPredTestByTrain = xTest * bufferValues.coeffsTrain;
        if (bufferValues.yPredTestByTest.size() == 0)
            bufferValues.yPredTestByTest = xTest * bufferValues.coeffsTest;
        return PairDVXd((bufferValues.yPredTestByTrain - bufferValues.yPredTestByTest).array().square().sum(), 
                         bufferValues.coeffsTrain);
    }

    PairDVXd Criterion::symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                           const VectorXd& yTest, BufferValues& bufferValues) const {
        if (bufferValues.coeffsTrain.size() == 0)
            bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (bufferValues.coeffsTest.size() == 0)
            bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (bufferValues.yPredTrainByTrain.size() == 0)
            bufferValues.yPredTrainByTrain = xTrain * bufferValues.coeffsTrain;
        if (bufferValues.yPredTrainByTest.size() == 0)
            bufferValues.yPredTrainByTest = xTrain * bufferValues.coeffsTest;
        if (bufferValues.yPredTestByTrain.size() == 0)
            bufferValues.yPredTestByTrain = xTest * bufferValues.coeffsTrain;
        if (bufferValues.yPredTestByTest.size() == 0)
            bufferValues.yPredTestByTest = xTest * bufferValues.coeffsTest;
        return PairDVXd((bufferValues.yPredTrainByTrain - bufferValues.yPredTrainByTest).array().square().sum() +
                        (bufferValues.yPredTestByTrain - bufferValues.yPredTestByTest).array().square().sum(), 
                         bufferValues.coeffsTrain);
    }

    PairDVXd Criterion::unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                       const VectorXd& yTest, BufferValues& bufferValues) const {
        if (bufferValues.coeffsTrain.size() == 0)
            bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (bufferValues.coeffsTest.size() == 0)
            bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
        return PairDVXd((bufferValues.coeffsTrain - bufferValues.coeffsTest).array().square().sum(), 
                         bufferValues.coeffsTrain);
    }

    PairDVXd Criterion::absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                          const VectorXd& yTest, BufferValues& bufferValues) const {
        VectorXd yPredTestByAll;
        if (bufferValues.coeffsTrain.size() == 0)
            bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (bufferValues.coeffsTest.size() == 0)
            bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (bufferValues.coeffsAll.size() == 0) {
            MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
            dataX << xTrain, xTest;
            VectorXd dataY(yTrain.size() + yTest.size());
            dataY << yTrain, yTest;
            bufferValues.coeffsAll = findBestCoeffs(dataX, dataY);
        }
        if (bufferValues.yPredTestByTrain.size() == 0)
            bufferValues.yPredTestByTrain = xTest * bufferValues.coeffsTrain;
        if (bufferValues.yPredTestByTest.size() == 0)
            bufferValues.yPredTestByTest = xTest * bufferValues.coeffsTest;
        yPredTestByAll = xTest * bufferValues.coeffsAll;
        return PairDVXd(((yPredTestByAll - bufferValues.yPredTestByTrain) * 
                         (bufferValues.yPredTestByTest - yPredTestByAll)).array().sum(), bufferValues.coeffsTrain);
    }

    PairDVXd Criterion::symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                             const VectorXd& yTest, BufferValues& bufferValues) const {
        VectorXd yPredAllByTrain, yPredAllByTest, yPredAllByAll;
        MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
        dataX << xTrain, xTest;
        VectorXd dataY(yTrain.size() + yTest.size());
        dataY << yTrain, yTest;

        if (bufferValues.coeffsTrain.size() == 0)
            bufferValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (bufferValues.coeffsTest.size() == 0)
            bufferValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (bufferValues.coeffsAll.size() == 0)
            bufferValues.coeffsAll = findBestCoeffs(dataX, dataY);
        
        yPredAllByTrain = dataX * bufferValues.coeffsTrain;
        yPredAllByTest = dataX * bufferValues.coeffsTest;
        yPredAllByAll = dataX * bufferValues.coeffsAll;
        return PairDVXd(((yPredAllByAll - yPredAllByTrain) * (yPredAllByTest - yPredAllByAll)).array().sum(), 
                         bufferValues.coeffsTrain);
    }

    PairDVXd Criterion::getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, 
                                  const VectorXd& yTest, CriterionType _criterionType, BufferValues& bufferValues) const {
        switch (_criterionType) {
        case CriterionType::regularity:
            return regularity(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::symRegularity:
            return symRegularity(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::stability:
            return stability(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::symStability:
            return symStability(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::unbiasedOutputs:
            return unbiasedOutputs(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::symUnbiasedOutputs:
            return symUnbiasedOutputs(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::unbiasedCoeffs:
            return unbiasedCoeffs(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::absoluteStability:
            return absoluteStability(xTrain, xTest, yTrain, yTest, bufferValues);
        case CriterionType::symAbsoluteStability:
            return symAbsoluteStability(xTrain, xTest, yTrain, yTest, bufferValues); // TODO: default return (because compilation warning)
        }
    }

    Criterion::Criterion(CriterionType _criterionType, Solver _solver) {
        criterionType = _criterionType;
        solver = _solver;
    }

    std::string Criterion::getClassName() const {
        std::string className = std::string(boost::typeindex::type_id_runtime(*this).pretty_name());
        className = className.substr(className.find_last_of(':') + 1);
        return className;
    }

    PairDVXd Criterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                  const VectorXd& yTest) const {
        BufferValues tempValues;
        return getResult(xTrain, xTest, yTrain, yTest, criterionType, tempValues);
    }

    ParallelCriterion::ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType,
        double _alpha, Solver _solver) : Criterion(_firstCriterionType, _solver) {
        secondCriterionType = _secondCriterionType;
        alpha = _alpha;
    }

    PairDVXd ParallelCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                          const VectorXd& yTest) const {
        BufferValues tempValues;
        PairDVXd firstResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType, tempValues);
        PairDVXd secondResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType, tempValues);
        return PairDVXd(alpha * firstResult.first + (1 - alpha) * secondResult.first, firstResult.second);
    }

    SequentialCriterion::SequentialCriterion(CriterionType _firstCriterionType, 
                                             CriterionType _secondCriterionType, Solver _solver)
        : Criterion(_firstCriterionType, _solver) {
        secondCriterionType = _secondCriterionType;
    }

    PairDVXd SequentialCriterion::recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain,
                                              const VectorXd& yTest, const VectorXd& _coeffsTrain) const {
        BufferValues tempValues;
        tempValues.coeffsTrain = _coeffsTrain;
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType, tempValues);
    }
}