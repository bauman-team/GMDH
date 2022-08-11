#include "criterion.h"

namespace GMDH {
class COMBI : public GMDH {

    std::vector<int> bestColsIndexes;
    VectorXd bestCoeffs;
    int inputColsNumber;

    std::vector<std::vector<bool>> getCombinations(int n_cols, int level) const;
    static std::vector<int> polynomialToIndexes(const std::vector<bool>& polynomial);
        
public:
    int save(const std::string& path) const override;
    int load(const std::string& path) override;
    double predict(const RowVectorXd& x) const override;
    VectorXd predict(const MatrixXd& x) const override;
    COMBI& fit(MatrixXd x, VectorXd y, const Criterion& criterion, int threads = 1, int verbose = 0) override;
    std::string getBestPolynomial() const override;
};
}