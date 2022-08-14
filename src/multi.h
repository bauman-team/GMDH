#include "combi.h"

namespace GMDH {

    class MULTI : public COMBI {
    protected:
        std::vector<std::vector<uint16_t>> getCombinations(int n_cols, int level) const override;
    };

}