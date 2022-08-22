#include "combi.h"

namespace GMDH {

    class MULTI : public COMBI {
    protected:
        VectorVu16 getCombinations(int n_cols, int level) const override;
    public:
        GMDH& fit(MatrixXd x, VectorXd y, Criterion& criterion, int _kBest, double testSize = 0.5, bool shuffle = false,
                int randomSeed = 0, uint8_t p = 1, int threads = 1, int verbose = 0);
    };
}