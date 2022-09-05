#include "gmdh.h"

namespace GMDH {

	enum class PolynomialType {linear, linear_cov, quadratic};

	class GMDH_API MIA : public GMDH {
	protected:
		PolynomialType polynomialType;

		VectorVu16 generateCombinations(int n_cols) const override;
		MatrixXd getPolynomialX(const MatrixXd& x) const;

		void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
								   IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const override;

		bool nextLevelCondition(double& lastLevelEvaluation, int kBest, uint8_t pAverage, VectorC& combinations,
							    const Criterion& criterion, SplittedData& data, double limit) override;

		virtual void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations);
		virtual void removeExtraCombinations();
		std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
		std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const override;

	public:
		GMDH& fit(MatrixXd x, VectorXd y, Criterion& criterion, int _kBest, 
				  PolynomialType _polynomialType = PolynomialType::quadratic, double testSize = 0.5, bool shuffle = false, 
				  int randomSeed = 0, uint8_t pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);

		int save(const std::string& path) const override;
		int load(const std::string& path) override;

		double predict(const RowVectorXd& x) const override;
		virtual VectorXd predict(const MatrixXd& x) const override;
	};
}