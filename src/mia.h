#include "gmdh.h"

namespace GMDH {

	enum class PolynomialType {linear, linear_cov, quadratic};

	class GMDH_API MIA : public GMDH {
	protected:
		PolynomialType polynomialType;

		VectorVu16 generateCombinations(int n_cols) const override;
		//MatrixXd polynomailFeatures(const MatrixXd& X, int max_degree);
		MatrixXd getPolynomialX(const MatrixXd& x) const;

		void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
								   IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const override;

		bool nextLevelCondition(double& lastLevelEvaluation, int kBest, uint8_t pAverage, VectorC& combinations,
							    const Criterion& criterion, SplittedData& data) override;
	public:
		GMDH& fit(MatrixXd x, VectorXd y, Criterion& criterion, int _kBest, 
				  PolynomialType _polynomialType = PolynomialType::quadratic, double testSize = 0.5, 
				  bool shuffle = false, int randomSeed = 0, uint8_t pAverage = 1, int threads = 1, int verbose = 0);

		int save(const std::string& path) const override;
		int load(const std::string& path) override;

		double predict(const RowVectorXd& x) const override;
		VectorXd predict(const MatrixXd& x) const override;
		std::string getBestPolynomial() const override;
	};
}