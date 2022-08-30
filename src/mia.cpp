#include "mia.h"

namespace GMDH {

	VectorVu16 MIA::generateCombinations(int n_cols) const {
		return nChooseK(n_cols, 2);
	}

    MatrixXd MIA::getPolynomialX(const MatrixXd& x) const {
        MatrixXd polyX(x);
        if ((polynomialType == PolynomialType::linear_cov)) {
            polyX.conservativeResize(NoChange, 4);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(2);
        }
        else if ((polynomialType == PolynomialType::quadratic)) {
            polyX.conservativeResize(NoChange, 6);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(0).cwiseProduct(x.col(0));
            polyX.col(4) = x.col(1).cwiseProduct(x.col(1));
            polyX.col(5) = x.col(2);
        }
        return polyX;
    }

    void MIA::polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
                                    IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const {
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

    bool MIA::nextLevelCondition(double& lastLevelEvaluation, int kBest, uint8_t pAverage, VectorC& combinations,
                                 const Criterion& criterion, SplittedData& data) {
        VectorC _bestCombinations = getBestCombinations(combinations, kBest);
        if (criterion.getClassName() == "SequentialCriterion") {
            // TODO: add threads or kBest value will be always small?
            for (auto combBegin = std::begin(_bestCombinations), 
                        combEnd = std::end(_bestCombinations); combBegin != combEnd; ++combBegin) {
                auto pairCoeffsEvaluation = static_cast<const SequentialCriterion&>(criterion).recalculate(
                    getPolynomialX(data.xTrain(Eigen::all, (*combBegin).combination())),
                    getPolynomialX(data.xTest(Eigen::all, (*combBegin).combination())),
                    data.yTrain, data.yTest, (*combBegin).bestCoeffs());
                (*combBegin).setEvaluation(pairCoeffsEvaluation.first);
            }
            std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        }
        double currLevelEvaluation = getMeanCriterionValue(_bestCombinations, pAverage);

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
            for (int i = 0; i < bestCombinations[level - 1].size(); ++i) {
                auto comb = bestCombinations[level - 1][i].combination();
                xTrainNew.col(i) = getPolynomialX(data.xTrain(Eigen::all, comb)) * bestCombinations[level - 1][i].bestCoeffs();
                xTestNew.col(i) = getPolynomialX(data.xTest(Eigen::all, comb)) * bestCombinations[level - 1][i].bestCoeffs();
            }
            xTrainNew.col(xTrainNew.cols() - 1) = VectorXd::Ones(xTrainNew.rows());
            xTestNew.col(xTestNew.cols() - 1) = VectorXd::Ones(xTestNew.rows());
            data.xTrain = xTrainNew;
            data.xTest = xTestNew;

            /*std::cout << "new_x_train_data:\n" << data.xTrain << "\n\n";
            std::cout << "new_x_test_data:\n" << data.xTest << "\n\n";
            std::cout << "new_y_train_data:\n" << data.yTrain << "\n\n";
            std::cout << "new_y_test_data:\n" << data.yTest << "\n\n";*/

            ++level;
            return true;
        }

        /*for (int l = 0; l < bestCombinations.size(); ++l) {
            std::cout << "LEVEL " << l + 1 << "\n\n";
            for (int i = 0; i < bestCombinations[l].size(); ++i) {
                for (int j = 0; j < bestCombinations[l][i].combination().size(); ++j)
                    std::cout << bestCombinations[l][i].combination()[j] << " ";
                std::cout << "\n" << bestCombinations[l][i].bestCoeffs() << "\n\n";
            }
        }*/

        std::vector<VectorC> realBestCombinations(bestCombinations.size());
        realBestCombinations[realBestCombinations.size() - 1] = VectorC(1, bestCombinations[level - 2][0]);
        for (int i = realBestCombinations.size() - 1; i > 0; --i) {
            std::set<uint16_t> usedCombinationsIndexes;
            for (int j = 0; j < realBestCombinations[i].size(); ++j) {
                auto comb = realBestCombinations[i][j].combination();
                for (int k = 0; k < comb.size() - 1; ++k)
                    usedCombinationsIndexes.insert(comb[k]);
            }
            for (auto it = usedCombinationsIndexes.begin(); it != usedCombinationsIndexes.end(); ++it)
                realBestCombinations[i - 1].push_back(bestCombinations[i - 1][*it]);
            for (int j = 0; j < realBestCombinations[i].size(); ++j) {
                auto comb = realBestCombinations[i][j].combination();
                for (int k = 0; k < comb.size() - 1; ++k)
                    comb[k] = std::distance(usedCombinationsIndexes.begin(), usedCombinationsIndexes.find(comb[k]));
                comb[comb.size() - 1] = usedCombinationsIndexes.size();
                realBestCombinations[i][j].setCombination(std::move(comb));
            }
        }
        bestCombinations = realBestCombinations;

        /*for (int l = 0; l < bestCombinations.size(); ++l) {
            std::cout << "LEVEL " << l + 1 << "\n\n";
            for (int i = 0; i < bestCombinations[l].size(); ++i) {
                for (int j = 0; j < bestCombinations[l][i].combination().size(); ++j)
                    std::cout << bestCombinations[l][i].combination()[j] << " ";
                std::cout << "\n" << bestCombinations[l][i].bestCoeffs() << "\n\n";
            }
        }*/
       
        return false;
    }

    GMDH& MIA::fit(MatrixXd x, VectorXd y, Criterion& criterion, int kBest, PolynomialType _polynomialType, 
                   double testSize, bool shuffle, int randomSeed, uint8_t pAverage, int threads, int verbose) {
        polynomialType = _polynomialType;
        return GMDH::fit(x, y, criterion, kBest, testSize, shuffle, randomSeed, pAverage, threads, verbose);
    }

    int MIA::save(const std::string& path) const {
        std::ofstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1;
        else {
            modelFile << getModelName() << "\n";
            modelFile << inputColsNumber << "\n";
            for (int i = 0; i < bestCombinations.size(); ++i) {
                modelFile << "~\n";
                for (int j = 0; j < bestCombinations[i].size(); ++j)
                    modelFile << bestCombinations[i][j].getInfoForSaving();
            }
            modelFile.close();
        }
        return 0;
    }

    int MIA::load(const std::string& path) {
        inputColsNumber = 0;
        bestCombinations.clear();

        std::ifstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1;
        else {
            std::string modelName;
            modelFile >> modelName;
            if (modelName != getModelName())
                return -1;
            else {
                (modelFile >> inputColsNumber).get();

                int currLevel = -1;
                while (modelFile.peek() != EOF) {

                    if (modelFile.peek() == '~') {
                        std::string buffer;
                        std::getline(modelFile, buffer);
                        ++currLevel;
                        bestCombinations.push_back(VectorC());
                    }

                    std::string colsIndexesLine;
                    VectorU16 bestColsIndexes;
                    std::getline(modelFile, colsIndexesLine);
                    std::stringstream indexStream(colsIndexesLine);
                    uint16_t index;
                    while (indexStream >> index)
                        bestColsIndexes.push_back(index);

                    std::string coeffsLine;
                    std::vector<double> coeffs;
                    std::getline(modelFile, coeffsLine);
                    std::stringstream coeffsStream(coeffsLine);
                    double coeff;
                    while (coeffsStream >> coeff)
                        coeffs.push_back(coeff);

                    bestCombinations[currLevel].push_back(Combination(std::move(bestColsIndexes), 
                                                          Map<VectorXd>(coeffs.data(), coeffs.size())));
                }
            }
            modelFile.close();
        }
        return 0;
    }

    double MIA::predict(const RowVectorXd& x) const {
        return predict(MatrixXd(x))[0];
    }

    VectorXd MIA::predict(const MatrixXd& x) const {
        MatrixXd modifiedX(x.rows(), x.cols() + 1);
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        for (int i = 0; i < bestCombinations.size(); ++i) {
            MatrixXd xNew(x.rows(), bestCombinations[i].size() + 1);
            for (int j = 0; j < bestCombinations[i].size(); ++j) {
                auto comb = bestCombinations[i][j].combination();
                auto coeffs = bestCombinations[i][j].bestCoeffs();
                auto poly = getPolynomialX(modifiedX(Eigen::all, comb));
                xNew.col(j) = poly * coeffs;
            }
            xNew.col(xNew.cols() - 1) = VectorXd::Ones(xNew.rows());
            modifiedX = xNew;
        }
        return modifiedX.col(0);
    }

    std::string MIA::getBestPolynomial() const {
        std::string polynomialStr = "";

        for (int i = 0; i < bestCombinations.size(); ++i) {
            //polynomialStr += "LEVEL " + std::to_string(i + 1) + ":\n";
            for (int j = 0; j < bestCombinations[i].size(); ++j) {

                if (i < bestCombinations.size() - 1)
                    polynomialStr += "f" + std::to_string(i + 1) + "_" + std::to_string(j + 1) + " =";
                else
                    polynomialStr += "y =";

                auto bestColsIndexes = bestCombinations[i][j].combination();
                auto bestCoeffs = bestCombinations[i][j].bestCoeffs();
                for (int k = 0; k < bestCoeffs.size(); ++k) {
                    if (bestCoeffs[k] > 0) {
                        if (k > 0)
                            polynomialStr += " + ";
                        else
                            polynomialStr += " ";
                    }
                    else
                        polynomialStr += " - ";
                    polynomialStr += std::to_string(abs(bestCoeffs[k]));
                    if (i == 0) {
                        if (k < 2)
                            polynomialStr += "*x" + std::to_string(bestColsIndexes[k] + 1);
                        else if (k == 2 && bestCoeffs.size() > 3)
                            polynomialStr += "*x" + std::to_string(bestColsIndexes[0] + 1) +
                                             "*x" + std::to_string(bestColsIndexes[1] + 1);
                        else if (k < 5 && bestCoeffs.size() > 4)
                            polynomialStr += "*x" + std::to_string(bestColsIndexes[k - 3] + 1) + "^2";
                    }
                    else {
                        if (k < 2)
                            polynomialStr += "*f" + std::to_string(i) + "_" + std::to_string(bestColsIndexes[k] + 1);
                        else if (k == 2 && bestCoeffs.size() > 3)
                            polynomialStr += "*f" + std::to_string(i) + "_" + std::to_string(bestColsIndexes[0] + 1) +
                                             "*f" + std::to_string(i) + "_" + std::to_string(bestColsIndexes[1] + 1);
                        else if (k < 5 && bestCoeffs.size() > 4)
                            polynomialStr += "*f" + std::to_string(i) + "_" + std::to_string(bestColsIndexes[k - 3] + 1) + "^2";
                    }
                }
                if (i < bestCombinations.size() - 1 || j < bestCombinations[i].size() - 1)
                    polynomialStr += "\n";
            }
            if (i < bestCombinations.size() - 1)
                polynomialStr += "\n";
        }
        return polynomialStr;
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