#pragma once
#include "mia.h"

namespace GMDH {

/// @brief Class implementing relaxation iterative RIA algorithm
class GMDH_API RIA : public MIA {
protected:
	VectorVu16 generateCombinations(int n_cols) const override;
	void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) override;
	void removeExtraCombinations() override;
	std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
	std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
										const VectorU16& bestColsIndexes) const override;
public:
	/// @copydoc MIA::fit
	GmdhModel& fit(const MatrixXd& x, const VectorXd& y,
		const Criterion& criterion = Criterion(CriterionType::regularity), int kBest = 1,
		PolynomialType _polynomialType = PolynomialType::quadratic, double testSize = 0.5,
		int pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);

	using GmdhModel::predict;	
	VectorXd predict(const MatrixXd& x) const override;
};
}