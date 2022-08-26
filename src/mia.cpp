#include "mia.h"

namespace GMDH {

	VectorVu16 MIA::getCombinations(int n) const
	{
		return nChooseK(n, 2);
	}

    MatrixXd MIA::getPolynomialX(const MatrixXd& x) const
    {
        MatrixXd polyX(x);
        if ((polynomialType == PolynomialType::linear_cov))
        {
            polyX.conservativeResize(NoChange, 4);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(2);
        }
        else if ((polynomialType == PolynomialType::quadratic))
        {
            polyX.conservativeResize(NoChange, 6);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(0).cwiseProduct(x.col(0));
            polyX.col(4) = x.col(1).cwiseProduct(x.col(1));
            polyX.col(5) = x.col(2);
        }
        return polyX;
    }

    void MIA::polinomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const
    {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            auto pairCoeffsEvaluation = criterion.calculate(
                                            getPolynomialX(data.xTrain(Eigen::all, (*beginCoeffsVec).combination())),
                                            getPolynomialX(data.xTest(Eigen::all, (*beginCoeffsVec).combination())),
                                            data.yTrain, data.yTest);
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
            if (unlikely(verbose))
                --(*leftTasks);
        }
    }

    bool MIA::nextLevelCondition(double& lastLevelEvaluation, int kBest, uint8_t p, VectorC& combinations, const Criterion& criterion, SplittedData& data)
    {
        VectorC _bestCombinations = getBestCombinations(combinations, kBest);
        if (criterion.getClassName() == "SequentialCriterion") {
            // TODO: add threads or kBest value will be always small?
            for (auto combBegin = std::begin(_bestCombinations), combEnd = std::end(_bestCombinations); combBegin != combEnd; ++combBegin) {
                auto pairCoeffsEvaluation = static_cast<const SequentialCriterion&>(criterion).recalculate(
                    getPolynomialX(data.xTrain(Eigen::all, (*combBegin).combination())),
                    getPolynomialX(data.xTest(Eigen::all, (*combBegin).combination())),
                    data.yTrain, data.yTest, (*combBegin).bestCoeffs());
                (*combBegin).setEvaluation(pairCoeffsEvaluation.first);
            }
            std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        }
        double currLevelEvaluation = getMeanCriterionValue(_bestCombinations, p);

        if (lastLevelEvaluation > currLevelEvaluation) {
            bestCombinations.push_back(std::move(_bestCombinations));
            lastLevelEvaluation = currLevelEvaluation;

            /*for (int i = 0; i < bestCombinations[level - 1].size(); ++i)
            {
                for (int j = 0; j < bestCombinations[level - 1][i].combination().size(); ++j)
                    std::cout << bestCombinations[level - 1][i].combination()[j] << " ";
                std::cout << "\n" << bestCombinations[level - 1][i].bestCoeffs() << "\n\n";
            }
            std::cout << "old_x_train_data:\n" << data.xTrain << "\n\n";
            std::cout << "old_x_test_data:\n" << data.xTest << "\n\n";
            std::cout << "old_y_train_data:\n" << data.yTrain << "\n\n";
            std::cout << "old_y_test_data:\n" << data.yTest << "\n\n";*/

            MatrixXd xTrainNew(data.xTrain.rows(), bestCombinations[level - 1].size() + 1);
            MatrixXd xTestNew(data.xTest.rows(), bestCombinations[level - 1].size() + 1);
            std::vector<bool> usedCombinations;
            /*if (level > 1)
                usedCombinations.resize(bestCombinations[level - 2].size(), false);*/
            for (int i = 0; i < bestCombinations[level - 1].size(); ++i)
            {
                auto comb = bestCombinations[level - 1][i].combination();
                /*if (level > 1)
                    for (int j = 0; j < comb.size() - 1; ++j)
                        usedCombinations[comb[j]] = true;*/
                xTrainNew.col(i) = getPolynomialX(data.xTrain(Eigen::all, comb)) * bestCombinations[level - 1][i].bestCoeffs();
                xTestNew.col(i) = getPolynomialX(data.xTest(Eigen::all, comb)) * bestCombinations[level - 1][i].bestCoeffs();
            }
            /*if (level > 1)
                for (int i = 0; i < bestCombinations[level - 2].size(); ++i)
                    if (!usedCombinations[i])
                        bestCombinations[level - 2].erase(bestCombinations[level - 2].begin() + i);*/
            xTrainNew.col(xTrainNew.cols() - 1) = VectorXd::Ones(xTrainNew.rows());
            xTestNew.col(xTestNew.cols() - 1) = VectorXd::Ones(xTestNew.rows());
            data.xTrain = xTrainNew;
            data.xTest = xTestNew;

            /*std::cout << "new_x_train_data:\n" << data.xTrain << "\n\n";
            std::cout << "new_x_test_data:\n" << data.xTest << "\n\n";
            std::cout << "new_y_train_data:\n" << data.yTrain << "\n\n";
            std::cout << "new_y_test_data:\n" << data.yTest << "\n\n";*/

            /*bestCombinations[level - 2] = 
            for (int i = bestCombinations.size() - 1; i >= 0; --i)
            {

            }*/

            ++level;
            return true;
        }

        return false;
    }

    GMDH& MIA::fit(MatrixXd x, VectorXd y, Criterion& criterion, int kBest, PolynomialType _polynomialType, double testSize, bool shuffle, int randomSeed, uint8_t p, int threads, int verbose)
    {
        polynomialType = _polynomialType;
        return GMDH::fit(x, y, criterion, kBest, testSize, shuffle, randomSeed, p, threads, verbose);
    }

    int MIA::save(const std::string& path) const
    {
        return 0;
    }

    int MIA::load(const std::string& path)
    {
        return 0;
    }

    double MIA::predict(const RowVectorXd& x) const
    {
        for (int i = 0; i < bestCombinations.size(); ++i)
        {
            for (int j = 0; j < bestCombinations[i].size(); ++i)
            {

            }
        }
        return 0.0;
    }

    VectorXd MIA::predict(const MatrixXd& x) const
    {
        return VectorXd();
    }

    std::string MIA::getBestPolynomial() const
    {
        return std::string();
    }

    /*MatrixXd MIA::polynomailFeatures(const MatrixXd& X, int max_degree) {
        int n = X.cols();
        std::vector <int> d(n);
        std::iota(d.begin(), d.end(), 0);
        MatrixXd poly_X;
        std::vector<std::vector<int>> monoms;
        for (int degree = 1; degree <= max_degree; ++degree)
        {
            std::vector <int> v(degree + 1, 0);
            while (1) {
                for (int i = 0; i < degree; i++) {
                    if (v[i] >= n) {
                        v[i + 1] += 1;
                        for (int k = i; k >= 0; k--) v[k] = v[i + 1];
                    }
                }
                if (v[degree] > 0) break;

                VectorXd column = VectorXd::Ones(X.rows());
                std::vector<int> monom;
                for (int i = degree - 1; i > -1; i--)
                {
                    column = column.cwiseProduct(X.col(d[v[i]]));
                    monom.push_back(d[v[i]]);
                }
                poly_X.conservativeResize(X.rows(), poly_X.cols() + 1);
                poly_X.col(poly_X.cols() - 1) = column;
                monoms.push_back(monom);
                v[0]++;
            }
        }
        return poly_X;
    }*/
}