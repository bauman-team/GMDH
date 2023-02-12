#include "linear_model.h"

namespace GMDH {

void LinearModel::removeExtraCombinations() {
    bestCombinations[0] = VectorC(1, bestCombinations[0][0]);
}

bool LinearModel::preparations(SplittedData& data, VectorC&& _bestCombinations) {
    bestCombinations[0] = std::move(_bestCombinations);
    return level + 1 < data.xTrain.cols();
}

MatrixXd LinearModel::xDataForCombination(const MatrixXd& x, const VectorU16& comb) const {
    return x(Eigen::all, comb);
} // LCOV_EXCL_LINE

std::string LinearModel::getPolynomialPrefix(int levelIndex, int combIndex) const {
    return "y =";
}

std::string LinearModel::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const {
    return ((coeffIndex != coeffsNumber - 1) ? "x" + std::to_string(bestColsIndexes[coeffIndex] + 1) : "");
} // LCOV_EXCL_LINE

LinearModel::LinearModel() {
    bestCombinations.resize(1);
}

VectorXd LinearModel::predict(const MatrixXd& x) const {
    checkMatrixColsNumber(x);
    MatrixXd modifiedX{ x.rows(), x.cols() + 1 };
    modifiedX.col(x.cols()).setOnes();
    modifiedX.leftCols(x.cols()) = x;
    return modifiedX(Eigen::all, bestCombinations[0][0].combination()) * bestCombinations[0][0].bestCoeffs();
}
}
