#include "mia.h"

namespace GMDH {
	class GMDH_API RIA : public MIA {
	protected:
		VectorVu16 generateCombinations(int n_cols) const override;
		void transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) override;
		void removeExtraCombinations() override;
		std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
		std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const override;
	public:
		using GmdhModel::predict;
		VectorXd predict(const MatrixXd& x) const override;
	};
}