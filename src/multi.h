#include "gmdh.h"

namespace GMDH {

    class GMDH_API MULTI : public GMDH {
    protected:
        VectorVu16 generateCombinations(int n_cols) const override;
    public:
        MULTI();
        int save(const std::string& path) const override;
        int load(const std::string& path) override;

        GMDH& fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int _kBest, double testSize = 0.5,
                  bool shuffle = false, int randomSeed = 0, uint8_t pAverage = 1, int threads = 1, int verbose = 0);

        double predict(const RowVectorXd& x) const override;
        VectorXd predict(const MatrixXd& x) const override;
        std::string getBestPolynomial() const override;
    };
}