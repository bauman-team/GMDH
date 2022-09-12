#include "multi.h"

namespace GMDH {

    void MULTI::removeExtraCombinations() {
        bestCombinations[0] = VectorC(1, bestCombinations[0][0]);
    }

    bool MULTI::preparations(SplittedData& data, VectorC& _bestCombinations) {
        bestCombinations[0] = std::move(_bestCombinations);
        if (level + 1 < data.xTrain.cols())
            return true;
    }

    MatrixXd MULTI::xDataForCombination(const MatrixXd& x, const VectorU16& comb) const {
        return x(Eigen::all, comb);
    }

    VectorVu16 MULTI::generateCombinations(int n_cols) const {
        VectorVu16 combs;
        if (level == 1)
            return nChooseK(n_cols, level);

        for (auto comb : bestCombinations[0]) {
            for (auto i = 0; i < n_cols; ++i) {
                auto temp{ comb.combination() };
                if (std::find(std::begin(temp), std::end(temp), i) == std::end(temp)) {
                    temp.push_back(i);
                    std::sort(std::begin(temp), std::end(temp));
                    if (std::find(std::begin(combs), std::end(combs), temp) == std::end(combs))
                        combs.push_back(std::move(temp));
                }
            }
        }
        return combs;
    }

    std::string MULTI::getPolynomialPrefix(int levelIndex, int combIndex) const {
        return "y =";
    }

    std::string MULTI::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const {
        return ((coeffIndex != coeffsNumber - 1) ? "*x" + std::to_string(bestColsIndexes[coeffIndex] + 1) : "");
    }

    MULTI::MULTI() {
        bestCombinations.resize(1);
    }

    GmdhModel& MULTI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, double testSize, 
                          uint8_t pAverage, int threads, int verbose, double limit) {
        validateInputData(&testSize, &pAverage, &threads, &kBest);
        return GmdhModel::fit(x, y, criterion, kBest, testSize, pAverage, threads, verbose, limit);
    }

    VectorXd MULTI::predict(const MatrixXd& x) const {
        if (inputColsNumber != x.cols())
            throw GmdhException(GMDHPREDICTEXCEPTIONMSG);
        MatrixXd modifiedX{ x.rows(), x.cols() + 1 };
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        return modifiedX(Eigen::all, bestCombinations[0][0].combination()) * bestCombinations[0][0].bestCoeffs();
    }
}
