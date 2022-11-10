#include "linear_model.h"

namespace GMDH {

class GMDH_API COMBI : public LinearModel {
protected:
    VectorVu16 generateCombinations(int n_cols) const override;
public:
    COMBI() : LinearModel() {}

    GmdhModel& fit(const MatrixXd& x, const VectorXd& y,
                    const Criterion& criterion = Criterion(CriterionType::regularity),
                    double testSize = 0.5, int pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);
};
}