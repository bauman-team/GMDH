#include "ria.h"

namespace GMDH {

	VectorVu16 RIA::generateCombinations(int n_cols) const {
		if (level == 1)
			return MIA::generateCombinations(n_cols);
		VectorVu16 combs;
		for (uint16_t i = 0; i < inputColsNumber; ++i)
			for (uint16_t j = inputColsNumber; j < n_cols; ++j)
				combs.push_back(VectorU16{i, j, static_cast<uint16_t>(n_cols)});
		return combs;
	}

	void RIA::transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) {
		if (level == 1) {
			data.xTrain.conservativeResize(NoChange, data.xTrain.cols() + bestCombinations.size());
			data.xTest.conservativeResize(NoChange, data.xTest.cols() + bestCombinations.size());
			data.xTrain.col(data.xTrain.cols() - 1) = VectorXd::Ones(data.xTrain.rows());
			data.xTest.col(data.xTrain.cols() - 1) = VectorXd::Ones(data.xTest.rows());
		}
		MatrixXd newColsTrain(data.xTrain.rows(), bestCombinations.size());
		MatrixXd newColsTest(data.xTest.rows(), bestCombinations.size());
		for (int i = 0; i < bestCombinations.size(); ++i) {
			auto comb = bestCombinations[i].combination();
			newColsTrain.col(i) = getPolynomialX(data.xTrain(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
			newColsTest.col(i) = getPolynomialX(data.xTest(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
		}
		for (int i = 0; i < bestCombinations.size(); ++i) {
			data.xTrain.col(data.xTrain.cols() - bestCombinations.size() + i - 1) = newColsTrain.col(i);
			data.xTest.col(data.xTest.cols() - bestCombinations.size() + i - 1) = newColsTest.col(i);
		}
	}

	void RIA::removeExtraCombinations() {
		std::vector<VectorC> realBestCombinations(bestCombinations.size());
		realBestCombinations[realBestCombinations.size() - 1] = VectorC(1, bestCombinations[level - 2][0]);
		for (int i = realBestCombinations.size() - 1; i > 0; --i)
			realBestCombinations[i - 1].push_back(
				bestCombinations[i - 1][realBestCombinations[i][0].combination()[1] - inputColsNumber]);
		for (int i = realBestCombinations.size() - 1; i > 0; --i) {
			auto comb = realBestCombinations[i][0].combination();
			comb[1] = inputColsNumber;
			comb[2] = inputColsNumber + 1;
			realBestCombinations[i][0].setCombination(comb);
		}
		bestCombinations = realBestCombinations;
	}

	VectorXd RIA::predict(const MatrixXd& x) const {
		MatrixXd modifiedX(x.rows(), x.cols() + 2);
		modifiedX.col(x.cols()).setOnes();
		modifiedX.col(x.cols() + 1).setOnes();
		modifiedX.leftCols(x.cols()) = x;
		for (int i = 0; i < bestCombinations.size(); ++i) {
			//std::cout << modifiedX << std::endl;
			//std::cout << bestCombinations[i][0].bestCoeffs() << std::endl;
			auto comb = bestCombinations[i][0].combination();
			auto coeffs = bestCombinations[i][0].bestCoeffs();
			auto poly = getPolynomialX(modifiedX(Eigen::all, comb));
			//std::cout << poly << std::endl;
			modifiedX.col(x.cols()) = poly * coeffs;
			//std::cout << modifiedX.col(x.cols()) << std::endl;
		}
		return modifiedX.col(x.cols());
	}

	std::string RIA::getBestPolynomial() const {
		return std::string();
	}
}