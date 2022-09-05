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
                                 const Criterion& criterion, SplittedData& data, double limit) {
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
        //std::cout << "\n" << currLevelEvaluation << "\n";

        if (lastLevelEvaluation - currLevelEvaluation > limit) {
            bestCombinations.push_back(std::move(_bestCombinations));
            lastLevelEvaluation = currLevelEvaluation;
            transformDataForNextLevel(data, bestCombinations[level - 1]);
            ++level;
            return true;
        }
        removeExtraCombinations();
        return false;
    }

    void MIA::transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) {
        MatrixXd xTrainNew(data.xTrain.rows(), bestCombinations.size() + 1);
        MatrixXd xTestNew(data.xTest.rows(), bestCombinations.size() + 1);
        for (int i = 0; i < bestCombinations.size(); ++i) {
            auto comb = bestCombinations[i].combination();
            xTrainNew.col(i) = getPolynomialX(data.xTrain(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
            xTestNew.col(i) = getPolynomialX(data.xTest(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
        }
        xTrainNew.col(xTrainNew.cols() - 1) = VectorXd::Ones(xTrainNew.rows());
        xTestNew.col(xTestNew.cols() - 1) = VectorXd::Ones(xTestNew.rows());
        data.xTrain = xTrainNew;
        data.xTest = xTestNew;
    }

    void MIA::removeExtraCombinations() {
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
    }

    std::string MIA::getPolynomialPrefix(int levelIndex, int combIndex) const {
        return ((levelIndex < bestCombinations.size() - 1) ?
            "f" + std::to_string(levelIndex + 1) + "_" + std::to_string(combIndex + 1) : "y") + " =";
    }

    std::string MIA::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const {
        if (levelIndex == 0) {
            if (coeffIndex < 2)
                return "*x" + std::to_string(bestColsIndexes[coeffIndex] + 1);
            else if (coeffIndex == 2 && coeffsNumber > 3)
                return "*x" + std::to_string(bestColsIndexes[0] + 1) + "*x" + std::to_string(bestColsIndexes[1] + 1);
            else if (coeffIndex < 5 && coeffsNumber > 4)
                return "*x" + std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
            else return "";
        }
        else {
            if (coeffIndex < 2)
                return "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[coeffIndex] + 1);
            else if (coeffIndex == 2 && coeffsNumber > 3)
                return "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[0] + 1) +
                       "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[1] + 1);
            else if (coeffIndex < 5 && coeffsNumber > 4)
                return "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
            else return "";
        }
    }

    GmdhModel& MIA::fit(MatrixXd x, VectorXd y, Criterion& criterion, int kBest, PolynomialType _polynomialType,
                   double testSize, bool shuffle, int randomSeed, uint8_t pAverage, int threads, int verbose, double limit) {
        polynomialType = _polynomialType;
        return GmdhModel::fit(x, y, criterion, kBest, testSize, shuffle, randomSeed, pAverage, threads, verbose, limit);
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
        if (inputColsNumber != x.cols())
            throw GmdhException(GMDHPREDICTEXCEPTIONMSG);
        MatrixXd modifiedX(x.rows(), x.cols() + 1);
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        for (int i = 0; i < bestCombinations.size(); ++i) {
            MatrixXd xNew(x.rows(), bestCombinations[i].size() + 1);
            for (int j = 0; j < bestCombinations[i].size(); ++j) {
                auto comb = bestCombinations[i][j].combination();
                xNew.col(j) = getPolynomialX(modifiedX(Eigen::all, comb)) * bestCombinations[i][j].bestCoeffs();;
            }
            xNew.col(xNew.cols() - 1) = VectorXd::Ones(xNew.rows());
            modifiedX = xNew;
        }
        return modifiedX.col(0);
    }
}