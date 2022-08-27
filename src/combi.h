#include "multi.h"

namespace GMDH {

    class GMDH_API COMBI : public MULTI {
    protected:
        VectorVu16 getCombinations(int n_cols) const override;
    public:
        COMBI() : MULTI() {}
        GMDH& fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, double testSize = 0.5, bool shuffle = false,
            int randomSeed = 0, uint8_t p = 1, int threads = 1, int verbose = 0);
    };
}