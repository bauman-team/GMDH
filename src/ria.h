#include "mia.h"

namespace GMDH {
	class GMDH_API RIA : public MIA {
	protected:
		VectorVu16 generateCombinations(int n_cols) const override;
		void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) override;
		void removeExtraCombinations() override;
	public:
		VectorXd predict(const MatrixXd& x) const override;
		std::string getBestPolynomial() const override;
	};
}