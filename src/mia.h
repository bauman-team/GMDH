#include "gmdh.h"

namespace GMDH {

	enum class PolynomialType {linear, linear_cov, quadratic};

	class GMDH_API MIA : public GmdhModel {
	protected:
		PolynomialType polynomialType;

		VectorVu16 generateCombinations(int n_cols) const override;
		MatrixXd getPolynomialX(const MatrixXd& x) const;

		virtual void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations);
		virtual void removeExtraCombinations() override;
		virtual bool preparations(SplittedData& data, VectorC& _bestCombinations) override;

		virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const override;

		std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
		std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const override;

	public:
		GmdhModel& fit(MatrixXd x, VectorXd y, Criterion& criterion, int _kBest,
				  PolynomialType _polynomialType = PolynomialType::quadratic, double testSize = 0.5,
				  uint8_t pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);

		using GmdhModel::predict;
		virtual VectorXd predict(const MatrixXd& x) const override;
	};
}