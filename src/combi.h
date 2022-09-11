#include "multi.h"

namespace GMDH {

    class GMDH_API COMBI : public MULTI {
    protected:
        VectorVu16 generateCombinations(int n_cols) const override;
    public:
        COMBI() : MULTI() {}
        GmdhModel& fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion,
                  double testSize = 0.5, uint8_t pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);
    };
}