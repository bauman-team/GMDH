#include "multi.h"

namespace GMDH {

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

    GmdhModel& MULTI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int _kBest, double testSize, 
                          uint8_t pAverage, int threads, int verbose, double limit) { // TODO: whaaat? why not kBest without '_'
        validateInputData(&testSize, &pAverage, &threads, &_kBest);
        return GmdhModel::fit(x, y, criterion, _kBest, testSize, pAverage, threads, verbose, limit);
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
