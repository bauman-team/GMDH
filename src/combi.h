#include "gmdh.h"

namespace GMDH {

class COMBI : public GMDH {
protected:
    virtual VectorVu16 getCombinations(int n_cols, int level) const;
public:
    int save(const std::string& path) const override;
    int load(const std::string& path) override;

    GMDH& fit(MatrixXd x, VectorXd y, Criterion& criterion, double testSize = 0.5, bool shuffle = false,
        int randomSeed = 0, uint8_t p = 1, int threads = 1, int verbose = 0);

    double predict(const RowVectorXd& x) const override;
    VectorXd predict(const MatrixXd& x) const override;
    std::string getBestPolynomial() const override;
};
}