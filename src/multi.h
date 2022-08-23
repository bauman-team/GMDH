#include "combi.h"

namespace GMDH {

    class GMDH_API MULTI : public COMBI {
    protected:
        VectorVu16 getCombinations(int n_cols, int level) const override;
    };

}