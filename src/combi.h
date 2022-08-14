#include "gmdh.h"

namespace GMDH {

class COMBI : public GMDH {
protected:
    virtual std::vector<std::vector<uint16_t>> getCombinations(int n_cols, int level) const;
public:
    int save(const std::string& path) const override;
    int load(const std::string& path) override;
    double predict(const RowVectorXd& x) const override;
    VectorXd predict(const MatrixXd& x) const override;
    std::string getBestPolynomial() const override;
};
}