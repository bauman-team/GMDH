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
			data.xTest.col(data.xTest.cols() - 1) = VectorXd::Ones(data.xTest.rows());
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

	std::string RIA::getPolynomialPrefix(int levelIndex, int combIndex) const {
		return ((levelIndex < bestCombinations.size() - 1) ? "f" + std::to_string(levelIndex + 1) : "y") + " =";
	}

	std::string RIA::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
										   const VectorU16& bestColsIndexes) const {
		if (levelIndex == 0)
			return MIA::getPolynomialVariable(levelIndex, coeffIndex, coeffsNumber, bestColsIndexes);
		
		if (coeffIndex == 0)
			return "*x" + std::to_string(bestColsIndexes[coeffIndex] + 1);
		else if (coeffIndex == 1)
			return "*f" + std::to_string(levelIndex);
		else if (coeffIndex == 2 && coeffsNumber > 3)
			return "*x" + std::to_string(bestColsIndexes[0] + 1) + "*f" + std::to_string(levelIndex);
		else if (coeffIndex == 3 && coeffsNumber > 4)
			return "*x" + std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
		else if (coeffIndex == 4 && coeffsNumber > 4)
			return "*f" + std::to_string(levelIndex) + "^2";
		return "";
	}

	VectorXd RIA::predict(const MatrixXd& x) const {
		checkMatrixColsNumber(x);
		MatrixXd modifiedX(x.rows(), x.cols() + 2);
		modifiedX.col(x.cols()).setOnes();
		modifiedX.col(x.cols() + 1).setOnes();
		modifiedX.leftCols(x.cols()) = x;
		for (int i = 0; i < bestCombinations.size(); ++i) {
			auto comb = bestCombinations[i][0].combination();
			modifiedX.col(x.cols()) = getPolynomialX(modifiedX(Eigen::all, comb)) * bestCombinations[i][0].bestCoeffs();
		}
		return modifiedX.col(x.cols());
	}
}