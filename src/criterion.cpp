//#include "pch.h"
#include "gmdh.h"

namespace GMDH {

    VectorXd Criterion::findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const
    {
        if ((solver == Solver::accurate))
            return xTrain.fullPivHouseholderQr().solve(yTrain);
        else if ((solver == Solver::balanced))
            return xTrain.colPivHouseholderQr().solve(yTrain);
        else
            return xTrain.householderQr().solve(yTrain);
    }

    PairDVXd Criterion::regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues, bool inverseSplit) const
    {
        if (!inverseSplit) {
            if (tempValues.coeffsTrain.size() == 0)
                tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (tempValues.yPredTestByTrain.size() == 0)
                tempValues.yPredTestByTrain = xTest * tempValues.coeffsTrain;
            return PairDVXd((yTest - tempValues.yPredTestByTrain).array().square().sum(), tempValues.coeffsTrain);
        }
        else {
            if (tempValues.coeffsTest.size() == 0)
                tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
            if (tempValues.yPredTrainByTest.size() == 0)
                tempValues.yPredTrainByTest = xTrain * tempValues.coeffsTest;
            return PairDVXd((yTrain - tempValues.yPredTrainByTest).array().square().sum(), tempValues.coeffsTest);
        }
    }

    PairDVXd Criterion::symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        PairDVXd part1 = regularity(xTrain, xTest, yTrain, yTest, tempValues);
        PairDVXd part2 = regularity(xTrain, xTest, yTrain, yTest, tempValues, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues, bool inverseSplit) const
    {
        if (!inverseSplit) {
            if (tempValues.coeffsTrain.size() == 0)
                tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (tempValues.yPredTrainByTrain.size() == 0)
                tempValues.yPredTrainByTrain = xTrain * tempValues.coeffsTrain;
            if (tempValues.yPredTestByTrain.size() == 0)
                tempValues.yPredTestByTrain = xTest * tempValues.coeffsTrain;
            return PairDVXd((yTrain - tempValues.yPredTrainByTrain).array().square().sum() +
                            (yTest - tempValues.yPredTestByTrain).array().square().sum(), tempValues.coeffsTrain);
        }
        else {
            if (tempValues.coeffsTest.size() == 0)
                tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
            if (tempValues.yPredTrainByTest.size() == 0)
                tempValues.yPredTrainByTest = xTrain * tempValues.coeffsTest;
            if (tempValues.yPredTestByTest.size() == 0)
                tempValues.yPredTestByTest = xTest * tempValues.coeffsTest;
            return PairDVXd((yTrain - tempValues.yPredTrainByTest).array().square().sum() +
                            (yTest - tempValues.yPredTestByTest).array().square().sum(), tempValues.coeffsTest);
        }
    }

    PairDVXd Criterion::symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        PairDVXd part1 = stability(xTrain, xTest, yTrain, yTest, tempValues);
        PairDVXd part2 = stability(xTrain, xTest, yTrain, yTest, tempValues, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        if (tempValues.coeffsTrain.size() == 0)
            tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (tempValues.coeffsTest.size() == 0)
            tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (tempValues.yPredTestByTrain.size() == 0)
            tempValues.yPredTestByTrain = xTest * tempValues.coeffsTrain;
        if (tempValues.yPredTestByTest.size() == 0)
            tempValues.yPredTestByTest = xTest * tempValues.coeffsTest;
        return PairDVXd((tempValues.yPredTestByTrain - tempValues.yPredTestByTest).array().square().sum(), tempValues.coeffsTrain);
    }

    PairDVXd Criterion::symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        if (tempValues.coeffsTrain.size() == 0)
            tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (tempValues.coeffsTest.size() == 0)
            tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (tempValues.yPredTrainByTrain.size() == 0)
            tempValues.yPredTrainByTrain = xTrain * tempValues.coeffsTrain;
        if (tempValues.yPredTrainByTest.size() == 0)
            tempValues.yPredTrainByTest = xTrain * tempValues.coeffsTest;
        if (tempValues.yPredTestByTrain.size() == 0)
            tempValues.yPredTestByTrain = xTest * tempValues.coeffsTrain;
        if (tempValues.yPredTestByTest.size() == 0)
            tempValues.yPredTestByTest = xTest * tempValues.coeffsTest;
        return PairDVXd((tempValues.yPredTrainByTrain - tempValues.yPredTrainByTest).array().square().sum() +
                        (tempValues.yPredTestByTrain - tempValues.yPredTestByTest).array().square().sum(), tempValues.coeffsTrain);
    }

    PairDVXd Criterion::unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        if (tempValues.coeffsTrain.size() == 0)
            tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (tempValues.coeffsTest.size() == 0)
            tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
        return PairDVXd((tempValues.coeffsTrain - tempValues.coeffsTest).array().square().sum(), tempValues.coeffsTrain);
    }

    PairDVXd Criterion::absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        VectorXd yPredTestByAll;
        if (tempValues.coeffsTrain.size() == 0)
            tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (tempValues.coeffsTest.size() == 0)
            tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (tempValues.coeffsAll.size() == 0) {
            MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
            dataX << xTrain, xTest;
            VectorXd dataY(yTrain.size() + yTest.size());
            dataY << yTrain, yTest;
            tempValues.coeffsAll = findBestCoeffs(dataX, dataY);
        }
        if (tempValues.yPredTestByTrain.size() == 0)
            tempValues.yPredTestByTrain = xTest * tempValues.coeffsTrain;
        if (tempValues.yPredTestByTest.size() == 0)
            tempValues.yPredTestByTest = xTest * tempValues.coeffsTest;
        yPredTestByAll = xTest * tempValues.coeffsAll;
        return PairDVXd(((yPredTestByAll - tempValues.yPredTestByTrain) * (tempValues.yPredTestByTest - yPredTestByAll)).array().sum(), tempValues.coeffsTrain);
    }

    PairDVXd Criterion::symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, TempValues& tempValues) const
    {
        VectorXd yPredAllByTrain, yPredAllByTest, yPredAllByAll;
        MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
        dataX << xTrain, xTest;
        VectorXd dataY(yTrain.size() + yTest.size());
        dataY << yTrain, yTest;

        if (tempValues.coeffsTrain.size() == 0)
            tempValues.coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (tempValues.coeffsTest.size() == 0)
            tempValues.coeffsTest = findBestCoeffs(xTest, yTest);
        if (tempValues.coeffsAll.size() == 0)
            tempValues.coeffsAll = findBestCoeffs(dataX, dataY);
        
        yPredAllByTrain = dataX * tempValues.coeffsTrain;
        yPredAllByTest = dataX * tempValues.coeffsTest;
        yPredAllByAll = dataX * tempValues.coeffsAll;
        return PairDVXd(((yPredAllByAll - yPredAllByTrain) * (yPredAllByTest - yPredAllByAll)).array().sum(), tempValues.coeffsTrain);
    }

    PairDVXd Criterion::getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, CriterionType _criterionType, TempValues& tempValues) const
    {
        if ((_criterionType == CriterionType::regularity)) {
            return regularity(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::symRegularity)) {
            return symRegularity(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::stability)) {
            return stability(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::symStability)) {
            return symStability(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::unbiasedOutputs)) {
            return unbiasedOutputs(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::symUnbiasedOutputs)) {
            return symUnbiasedOutputs(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::unbiasedCoeffs)) {
            return unbiasedCoeffs(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else if ((_criterionType == CriterionType::absoluteStability)) {
            return absoluteStability(xTrain, xTest, yTrain, yTest, tempValues);
        }
        else {
            return symAbsoluteStability(xTrain, xTest, yTrain, yTest, tempValues);
        }
    }

    Criterion::Criterion(CriterionType _criterionType, Solver _solver)
    {
        criterionType = _criterionType;
        solver = _solver;
    }

    std::string Criterion::getClassName() const
    {
        std::string className = std::string(boost::typeindex::type_id_runtime(*this).pretty_name());
        className = className.substr(className.find_last_of(':') + 1);
        return className;
    }

    PairDVXd Criterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        TempValues tempValues;
        return getResult(xTrain, xTest, yTrain, yTest, criterionType, tempValues);
    }

    ParallelCriterion::ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType,
        double _alpha, Solver _solver) : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
        alpha = _alpha;
    }

    PairDVXd ParallelCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        TempValues tempValues;
        PairDVXd firstResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType, tempValues);
        PairDVXd secondResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType, tempValues);
        return PairDVXd(alpha * firstResult.first + (1 - alpha) * secondResult.first, firstResult.second);
    }

    SequentialCriterion::SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver)
        : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
    }

    PairDVXd SequentialCriterion::recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, const VectorXd& _coeffsTrain) const
    {
        TempValues tempValues;
        tempValues.coeffsTrain = _coeffsTrain;
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType, tempValues);
    }
}