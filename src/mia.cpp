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
    data.xTrain = std::move(xTrainNew);
    data.xTest = std::move(xTestNew);
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

bool MIA::preparations(SplittedData& data, VectorC&& _bestCombinations) {
    bestCombinations.push_back(std::move(_bestCombinations));
    transformDataForNextLevel(data, bestCombinations[level - 1]);
    return true;
}

MatrixXd MIA::xDataForCombination(const MatrixXd& x, const VectorU16& comb) const {
    return getPolynomialX(x(Eigen::all, comb));
}

std::string MIA::getPolynomialPrefix(int levelIndex, int combIndex) const {
    return ((levelIndex < bestCombinations.size() - 1) ?
        "f" + std::to_string(levelIndex + 1) + "_" + std::to_string(combIndex + 1) : "y") + " =";
}

std::string MIA::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                        const VectorU16& bestColsIndexes) const {
    if (levelIndex == 0) {
        if (coeffIndex < 2)
            return "x" + std::to_string(bestColsIndexes[coeffIndex] + 1);
        else if (coeffIndex == 2 && coeffsNumber > 3)
            return "x" + std::to_string(bestColsIndexes[0] + 1) + 
                    "*x" + std::to_string(bestColsIndexes[1] + 1);
        else if (coeffIndex < 5 && coeffsNumber > 4)
            return "x" + std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
    }
    else {
        if (coeffIndex < 2)
            return "f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[coeffIndex] + 1);
        else if (coeffIndex == 2 && coeffsNumber > 3)
            return "f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[0] + 1) +
                    "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[1] + 1);
        else if (coeffIndex < 5 && coeffsNumber > 4)
            return "f" + std::to_string(levelIndex) + "_" + 
                    std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
    }
    return "";
}

boost::json::value MIA::toJSON() const {
    return boost::json::object{
        {"modelName", getModelName()},
        {"inputColsNumber", inputColsNumber},
        {"polynomialType", static_cast<int>(polynomialType)},
        {"bestCombinations", bestCombinations},
    };
}

int MIA::fromJSON(boost::json::value jsonModel) {
    auto errorCode = GmdhModel::fromJSON(jsonModel);
    if (errorCode)
        return errorCode;
    auto& o = jsonModel.as_object();
    polynomialType = static_cast<PolynomialType>(o.at("polynomialType").as_int64());
    return 0;
}

GmdhModel& MIA::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
                    PolynomialType _polynomialType, double testSize, int pAverage, 
                    int threads, int verbose, double limit) {
    /*
    It is necessasy for kBest value to be >= 3 for the MIA algorithm because 
    the number of combinations at each level is equal to combinations of 2 elements from kBest.
    If kBest == 2 then there will be only one possible combination.
    If kBest == 1 then it will be impossible to create combinations at all.
    */
    if (kBest < 3) {
        std::string errorMsg = getVariableName("kBest", "k_best") + " value must be an integer >= 3";
        throw std::invalid_argument(errorMsg);
    }

    /*
    It is necessasy for x matrix to have 3 or more columns.
    if columns < 3 then the number of combinations won't be enough
    to move to the next levels.
    */
    if (x.cols() < 3) {
        std::string errorMsg = getVariableName("x", "X") + " columns must be >= 3";
        throw std::invalid_argument(errorMsg);
    }

    validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest);
    polynomialType = _polynomialType;
    return GmdhModel::gmdhFit(x, y, criterion, kBest, testSize, pAverage, threads, verbose, limit);
}

VectorXd MIA::predict(const MatrixXd& x) const {
    checkMatrixColsNumber(x);
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
