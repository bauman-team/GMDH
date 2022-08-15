#include "combi.h"

namespace GMDH {

    class MULTI : public COMBI {
    protected:
        VectorVu16 getCombinations(int n_cols, int level) const override;
    };

}