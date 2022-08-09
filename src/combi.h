#include "criterion.h"

namespace GMDH {
class COMBI : public GMDH {

    std::vector<int> best_cols_index;
    VectorXd best_coeffs;
    int input_cols_number;

    std::vector<std::vector<bool>> getCombinations(int n, int k) const;
    std::vector<int> polinomToIndexes(const std::vector<bool>& polinom) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    int save(const std::string& path) const override;
    int load(const std::string& path) override;
    double predict(const RowVectorXd& x) const override;
    VectorXd predict(const MatrixXd& x) const override;
    COMBI& fit(MatrixXd x, VectorXd y, const Criterion& criterion, int threads = 1, int verbose = 0) override;
    std::string getBestPolymon() const override;
};
}