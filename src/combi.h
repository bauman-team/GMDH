#include "criterion.h"

namespace GMDH {
class COMBI : public GMDH { // TODO: split into separate files

    std::vector<int> best_cols_index;
    VectorXd best_coeffs;
    int input_cols_number;

    std::vector<std::vector<bool>> getCombinations(int n, int k) const;
    std::vector<int> polinomToIndexes(std::vector<bool> polinom) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    COMBI();
    int save(std::string path) const override;
    int load(const std::string& path) override;
    double predict(RowVectorXd x) const override;
    VectorXd predict(MatrixXd x) const override;
    COMBI& fit(MatrixXd x, VectorXd y, const Criterion& criterion) override;
    std::string getBestPolymon() const override;
};
}