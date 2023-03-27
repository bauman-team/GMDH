#pragma once
#include "gmdh.h"

namespace GMDH {

/// @brief Enum class for specifying the polynomial type to be used to construct new variables from existing ones
enum class PolynomialType {
	linear, //!< \f$ f(x_i, x_j)=w_0+w_1x_i+w_2x_j \f$
	linear_cov, //!< \f$ f(x_i, x_j)=w_0+w_1x_i+w_2x_j+w_{12}x_ix_j \f$
	quadratic //!< \f$ f(x_i, x_j)=w_0+w_1x_i+w_2x_j+w_{12}x_ix_j+w_{11}x_i^2+w_{22}x_j^2 \f$ 
};

/// @brief Class implementing multilayered iterative MIA algorithm
class GMDH_API MIA : public GmdhModel {
protected:
	PolynomialType polynomialType; //!< Selected polynomial type

	VectorVu16 generateCombinations(int n_cols) const override;

	/**
	 * @brief Construct vector of the new variable values according to the selected polynomial type
	 * 
	 * @param x Matrix of input variables values for the selected polynomial type
	 * @return Construct vector of the new variable values
	 */
	MatrixXd getPolynomialX(const MatrixXd& x) const;

	/**
	 * @brief Transform data in the current training level by constructing new variables using selected polynomial type
	 * 
	 * @param data Data used to train models at the current level
	 * @param bestCombinations Vector of the k best models of the current level
	 */
	virtual void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations);

	virtual void removeExtraCombinations() override;
	virtual bool preparations(SplittedData& data, VectorC&& _bestCombinations) override;
	virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const override;

	std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
	std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
										const VectorU16& bestColsIndexes) const override;

	boost::json::value toJSON() const override;
    int fromJSON(boost::json::value jsonModel) override;

public:
	 /**
     * @brief Fit the algorithm to find the best solution
     * 
     * @param x Matrix of input data containing predictive variables
     * @param y Vector of the taget values for the corresponding x data
     * @param criterion Selected external criterion
     * @param kBest The number of best models based of which new models of the next level will be constructed
	 * @param _polynomialType Selected polynomial type to be used to construct new variables from existing ones during training
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
					const Criterion& criterion = Criterion(CriterionType::regularity), int kBest = 3,
					PolynomialType _polynomialType = PolynomialType::quadratic, double testSize = 0.5,
					int pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);

	using GmdhModel::predict;
	virtual VectorXd predict(const MatrixXd& x) const override;
};
}