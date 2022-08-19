#include "gmdh.h"

namespace GMDH {

    void Criterion::resetCoeffsAndYPred()
    {
        /*coeffsTrain.resize(0);
        coeffsTest.resize(0);
        coeffsAll.resize(0);
        yPredTrainByTrain.resize(0);
        yPredTrainByTest.resize(0);
        yPredTestByTrain.resize(0);
        yPredTestByTest.resize(0);*/

        coeffsTrain = VectorXd();
        coeffsTest = VectorXd();
        coeffsAll = VectorXd();
        if (yPredTrainByTrain.size() != 0)
            yPredTrainByTrain = VectorXd();
        if (yPredTrainByTest.size() != 0)
            yPredTrainByTest = VectorXd();
        if (yPredTestByTrain.size() != 0)
            yPredTestByTrain = VectorXd();
        if (yPredTestByTest.size() != 0)
            yPredTestByTest = VectorXd();
    }

    VectorXd Criterion::findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const
    {
        if ((solver == Solver::accurate))
            return xTrain.fullPivHouseholderQr().solve(yTrain);
        else if ((solver == Solver::balanced))
            return xTrain.colPivHouseholderQr().solve(yTrain);
        else
            return xTrain.householderQr().solve(yTrain);
    }

    PairDVXd Criterion::regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, bool inverseSplit)
    {
        if (!inverseSplit) {
            if (coeffsTrain.size() == 0)
                coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (yPredTestByTrain.size() == 0)
                yPredTestByTrain = xTest * coeffsTrain;
            return PairDVXd((yTest - yPredTestByTrain).array().square().sum(), coeffsTrain);
        }
        else {
            if (coeffsTest.size() == 0)
                coeffsTest = findBestCoeffs(xTest, yTest);
            if (yPredTrainByTest.size() == 0)
                yPredTrainByTest = xTrain * coeffsTest;
            return PairDVXd((yTrain - yPredTrainByTest).array().square().sum(), coeffsTest);
        }
    }

    PairDVXd Criterion::symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        PairDVXd part1 = regularity(xTrain, xTest, yTrain, yTest);
        PairDVXd part2 = regularity(xTrain, xTest, yTrain, yTest, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, bool inverseSplit)
    {
        if (!inverseSplit) {
            if (coeffsTrain.size() == 0)
                coeffsTrain = findBestCoeffs(xTrain, yTrain);
            if (yPredTrainByTrain.size() == 0)
                yPredTrainByTrain = xTrain * coeffsTrain;
            if (yPredTestByTrain.size() == 0)
                yPredTestByTrain = xTest * coeffsTrain;
            return PairDVXd((yTrain - yPredTrainByTrain).array().square().sum() +
                            (yTest - yPredTestByTrain).array().square().sum(), coeffsTrain);
        }
        else {
            if (coeffsTest.size() == 0)
                coeffsTest = findBestCoeffs(xTest, yTest);
            if (yPredTrainByTest.size() == 0)
                yPredTrainByTest = xTrain * coeffsTest;
            if (yPredTestByTest.size() == 0)
                yPredTestByTest = xTest * coeffsTest;
            return PairDVXd((yTrain - yPredTrainByTest).array().square().sum() +
                            (yTest - yPredTestByTest).array().square().sum(), coeffsTest);
        }
    }

    PairDVXd Criterion::symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        PairDVXd part1 = stability(xTrain, xTest, yTrain, yTest);
        PairDVXd part2 = stability(xTrain, xTest, yTrain, yTest, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        if (coeffsTrain.size() == 0)
            coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (coeffsTest.size() == 0)
            coeffsTest = findBestCoeffs(xTest, yTest);
        if (yPredTestByTrain.size() == 0)
            yPredTestByTrain = xTest * coeffsTrain;
        if (yPredTestByTest.size() == 0)
            yPredTestByTest = xTest * coeffsTest;
        return PairDVXd((yPredTestByTrain - yPredTestByTest).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        if (coeffsTrain.size() == 0)
            coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (coeffsTest.size() == 0)
            coeffsTest = findBestCoeffs(xTest, yTest);
        if (yPredTrainByTrain.size() == 0)
            yPredTrainByTrain = xTrain * coeffsTrain;
        if (yPredTrainByTest.size() == 0)
            yPredTrainByTest = xTrain * coeffsTest;
        if (yPredTestByTrain.size() == 0)
            yPredTestByTrain = xTest * coeffsTrain;
        if (yPredTestByTest.size() == 0)
            yPredTestByTest = xTest * coeffsTest;
        return PairDVXd((yPredTrainByTrain - yPredTrainByTest).array().square().sum() +
                        (yPredTestByTrain - yPredTestByTest).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        if (coeffsTrain.size() == 0)
            coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (coeffsTest.size() == 0)
            coeffsTest = findBestCoeffs(xTest, yTest);
        return PairDVXd((coeffsTrain - coeffsTest).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        VectorXd yPredTestByAll;
        if (coeffsTrain.size() == 0)
            coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (coeffsTest.size() == 0)
            coeffsTest = findBestCoeffs(xTest, yTest);
        if (coeffsAll.size() == 0) {
            MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
            dataX << xTrain, xTest;
            VectorXd dataY(yTrain.size() + yTest.size());
            dataY << yTrain, yTest;
            coeffsAll = findBestCoeffs(dataX, dataY);
        }
        if (yPredTestByTrain.size() == 0)
            yPredTestByTrain = xTest * coeffsTrain;
        if (yPredTestByTest.size() == 0)
            yPredTestByTest = xTest * coeffsTest;
        yPredTestByAll = xTest * coeffsAll;
        return PairDVXd(((yPredTestByAll - yPredTestByTrain) * (yPredTestByTest - yPredTestByAll)).array().sum(), coeffsTrain);
    }

    PairDVXd Criterion::symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        VectorXd yPredAllByTrain, yPredAllByTest, yPredAllByAll;
        MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
        dataX << xTrain, xTest;
        VectorXd dataY(yTrain.size() + yTest.size());
        dataY << yTrain, yTest;

        if (coeffsTrain.size() == 0)
            coeffsTrain = findBestCoeffs(xTrain, yTrain);
        if (coeffsTest.size() == 0)
            coeffsTest = findBestCoeffs(xTest, yTest);
        if (coeffsAll.size() == 0)
            coeffsAll = findBestCoeffs(dataX, dataY);
        
        yPredAllByTrain = dataX * coeffsTrain;
        yPredAllByTest = dataX * coeffsTest;
        yPredAllByAll = dataX * coeffsAll;
        return PairDVXd(((yPredAllByAll - yPredAllByTrain) * (yPredAllByTest - yPredAllByAll)).array().sum(), coeffsTrain);
    }

    PairDVXd Criterion::getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, CriterionType _criterionType)
    {
        if ((_criterionType == CriterionType::regularity)) {
            return regularity(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symRegularity)) {
            return symRegularity(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::stability)) {
            return stability(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symStability)) {
            return symStability(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::unbiasedOutputs)) {
            return unbiasedOutputs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symUnbiasedOutputs)) {
            return symUnbiasedOutputs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::unbiasedCoeffs)) {
            return unbiasedCoeffs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::absoluteStability)) {
            return absoluteStability(xTrain, xTest, yTrain, yTest);
        }
        else {
            return symAbsoluteStability(xTrain, xTest, yTrain, yTest);
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

    PairDVXd Criterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        resetCoeffsAndYPred();
        return getResult(xTrain, xTest, yTrain, yTest, criterionType);
    }

    ParallelCriterion::ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType,
        double _alpha, Solver _solver) : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
        alpha = _alpha;
    }

    PairDVXd ParallelCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        PairDVXd firstResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType);
        PairDVXd secondResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType);
        resetCoeffsAndYPred();
        return PairDVXd(alpha * firstResult.first + (1 - alpha) * secondResult.first, firstResult.second);
    }

    SequentialCriterion::SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver)
        : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
    }

    PairDVXd SequentialCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType);
    }

    PairDVXd SequentialCriterion::recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest)
    {
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType);
    }
}