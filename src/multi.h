#include "linear_model.h"

namespace GMDH {

    class GMDH_API MULTI : public LinearModel {
    protected:
        VectorVu16 generateCombinations(int n_cols) const override;
    public:
        MULTI() : LinearModel() {}

        GmdhModel& fit(const MatrixXd& x, const VectorXd& y,
            const Criterion& criterion = Criterion(CriterionType::regularity), int kBest = 3,
            double testSize = 0.5, uint8_t pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);
    };
}