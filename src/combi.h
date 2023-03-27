#pragma once
#include "linear_model.h"

namespace GMDH {

/// @brief Class implementing combinatorial COMBI algorithm
class GMDH_API COMBI : public LinearModel {
protected:
    VectorVu16 generateCombinations(int n_cols) const override;
public:
    /// @brief Construct a new COMBI object
    COMBI() : LinearModel() {}

     /**
     * @brief Fit the algorithm to find the best solution
     * 
     * @param x Matrix of input data containing predictive variables
     * @param y Vector of the taget values for the corresponding x data
     * @param criterion Selected external criterion
     * @param testSize Fraction of the input data that should be used to evaluate models at each level
     * @param pAverage The number of best models based of which the external criterion for each level will be calculated
     * @param threads The number of threads used for calculations. Set -1 to use max possible threads 
     * @param verbose 1 if the printing detailed infomation about training process is needed, otherwise 0
     * @param limit The minimum value by which the external criterion should be improved in order to continue training
     * @throw std::invalid_argument
     * @warning If the threads or verbose value is incorrect an exception won't be thrown. 
     * Insted, the incorrect value will be replaced with the default value and a corresponding warning will be displayed
     * @return A reference to the algorithm object for which the training was performed
     */
    GmdhModel& fit(const MatrixXd& x, const VectorXd& y,
                    const Criterion& criterion = Criterion(CriterionType::regularity),
                    double testSize = 0.5, int pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);
};
}