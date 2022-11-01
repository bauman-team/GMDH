#include "multi.h"

namespace GMDH
{
    VectorVu16 MULTI::generateCombinations(int n_cols) const { // TODO: maybe change for bit masks 
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

    GmdhModel& MULTI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, double testSize,
        int pAverage, int threads, int verbose, double limit) {
        validateInputData(&testSize, &pAverage, &threads, &verbose, &limit, &kBest);

        return GmdhModel::gmdhFit(x, y, criterion, kBest, testSize, pAverage, threads, verbose, limit);
    }
}