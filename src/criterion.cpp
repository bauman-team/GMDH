#include "gmdh.h"

namespace GMDH {

    VectorXd Criterion::findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const
    {
        VectorXd coeffs;
        switch (solver) {
        case Solver::accurate:
            coeffs = xTrain.fullPivHouseholderQr().solve(yTrain);
            break;
        case Solver::balanced:
            coeffs = xTrain.colPivHouseholderQr().solve(yTrain);
            break;
        case Solver::fast:
            coeffs = xTrain.householderQr().solve(yTrain);
            break;
        }
        return coeffs;
    }

    PairDVXd Criterion::regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, bool inverseSplit) const
    {
        VectorXd coeffs;
        VectorXd yPred;
        if (!inverseSplit)
        {
            coeffs = findBestCoeffs(xTrain, yTrain);
            yPred = xTest * coeffs;
            return PairDVXd((yTest - yPred).array().square().sum(), coeffs);
        }
        else
        {
            coeffs = findBestCoeffs(xTest, yTest);
            yPred = xTrain * coeffs;
            return PairDVXd((yTrain - yPred).array().square().sum(), coeffs);
        }
    }

    PairDVXd Criterion::symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        PairDVXd part1 = regularity(xTrain, xTest, yTrain, yTest);
        PairDVXd part2 = regularity(xTrain, xTest, yTrain, yTest, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, bool inverseSplit) const
    {
        VectorXd coeffs;
        VectorXd yPredTrain, yPredTest;
        if (!inverseSplit)
        {
            coeffs = findBestCoeffs(xTrain, yTrain);
            yPredTrain = xTrain * coeffs;
            yPredTest = xTest * coeffs;
            return PairDVXd((yTrain - yPredTrain).array().square().sum() +
                (yTest - yPredTest).array().square().sum(), coeffs);
        }
        else
        {
            coeffs = findBestCoeffs(xTest, yTest);
            yPredTrain = xTrain * coeffs;
            yPredTest = xTest * coeffs;
            return PairDVXd((yTrain - yPredTrain).array().square().sum() +
                (yTest - yPredTest).array().square().sum(), coeffs);
        }
    }

    PairDVXd Criterion::symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        PairDVXd part1 = stability(xTrain, xTest, yTrain, yTest);
        PairDVXd part2 = stability(xTrain, xTest, yTrain, yTest, true);
        return PairDVXd(part1.first + part2.first, part1.second);
    }

    PairDVXd Criterion::unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        VectorXd coeffsTrain, coeffsTest;
        VectorXd yPredTest1, yPredTest2;
        coeffsTrain = findBestCoeffs(xTrain, yTrain);
        coeffsTest = findBestCoeffs(xTest, yTest);
        yPredTest1 = xTest * coeffsTrain;
        yPredTest2 = xTest * coeffsTest;
        return PairDVXd((yPredTest1 - yPredTest2).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        VectorXd coeffsTrain, coeffsTest;
        VectorXd yPredTrain1, yPredTrain2, yPredTest1, yPredTest2;
        coeffsTrain = findBestCoeffs(xTrain, yTrain);
        coeffsTest = findBestCoeffs(xTest, yTest);
        yPredTrain1 = xTrain * coeffsTrain;
        yPredTrain2 = xTrain * coeffsTest;
        yPredTest1 = xTest * coeffsTrain;
        yPredTest2 = xTest * coeffsTest;
        return PairDVXd((yPredTrain1 - yPredTrain2).array().square().sum() +
            (yPredTest1 - yPredTest2).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        VectorXd coeffsTrain, coeffsTest;
        coeffsTrain = findBestCoeffs(xTrain, yTrain);
        coeffsTest = findBestCoeffs(xTest, yTest);
        return PairDVXd((coeffsTrain - coeffsTest).array().square().sum(), coeffsTrain);
    }

    PairDVXd Criterion::absoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        VectorXd coeffsTrain, coeffsTest, coeffsAll;
        VectorXd yPredTest1, yPredTest2, yPredTest3;

        MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
        dataX << xTrain, xTest;

        VectorXd dataY(yTrain.size() + yTest.size());
        dataY << yTrain, yTest;

        coeffsTrain = findBestCoeffs(xTrain, yTrain);
        coeffsTest = findBestCoeffs(xTest, yTest);
        coeffsAll = findBestCoeffs(dataX, dataY);
        yPredTest1 = xTest * coeffsTrain;
        yPredTest2 = xTest * coeffsTest;
        yPredTest3 = xTest * coeffsAll;
        return PairDVXd(((yPredTest3 - yPredTest1) * (yPredTest2 - yPredTest3)).array().sum(), coeffsTrain);
    }

    PairDVXd Criterion::symAbsoluteStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        VectorXd coeffsTrain, coeffsTest, coeffsAll;
        VectorXd yPredAll1, yPredAll2, yPredAll3;

        MatrixXd dataX(xTrain.rows() + xTest.rows(), xTrain.cols());
        dataX << xTrain, xTest;

        VectorXd dataY(yTrain.size() + yTest.size());
        dataY << yTrain, yTest;

        coeffsTrain = findBestCoeffs(xTrain, yTrain);
        coeffsTest = findBestCoeffs(xTest, yTest);
        coeffsAll = findBestCoeffs(dataX, dataY);
        yPredAll1 = dataX * coeffsTrain;
        yPredAll2 = dataX * coeffsTest;
        yPredAll3 = dataX * coeffsAll;
        return PairDVXd(((yPredAll3 - yPredAll1) * (yPredAll2 - yPredAll3)).array().sum(), coeffsTrain);
    }

    PairDVXd Criterion::getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, CriterionType _criterionType) const
    {
        if ((_criterionType == CriterionType::regularity))
        {
            return regularity(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symRegularity))
        {
            return symRegularity(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::stability))
        {
            return stability(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symStability))
        {
            return symStability(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::unbiasedOutputs))
        {
            return unbiasedOutputs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::symUnbiasedOutputs))
        {
            return symUnbiasedOutputs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::unbiasedCoeffs))
        {
            return unbiasedCoeffs(xTrain, xTest, yTrain, yTest);
        }
        else if ((_criterionType == CriterionType::absoluteStability))
        {
            return absoluteStability(xTrain, xTest, yTrain, yTest);
        }
        else
        {
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

    PairDVXd Criterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        return getResult(xTrain, xTest, yTrain, yTest, criterionType);
    }

    ParallelCriterion::ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType,
        double _alpha, Solver _solver) : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
        alpha = _alpha;
    }

    PairDVXd ParallelCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        PairDVXd firstResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType);
        PairDVXd secondResult = Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType);
        return PairDVXd(alpha * firstResult.first + (1 - alpha) * secondResult.first, firstResult.second);
    }

    SequentialCriterion::SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, Solver _solver)
        : Criterion(_firstCriterionType, _solver)
    {
        secondCriterionType = _secondCriterionType;
    }

    PairDVXd SequentialCriterion::calculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, criterionType);
    }

    PairDVXd SequentialCriterion::recalculate(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest) const
    {
        return Criterion::getResult(xTrain, xTest, yTrain, yTest, secondCriterionType);
    }
}