#include "mia.h"

namespace GMDH {

	VectorVu16 MIA::generateCombinations(int n_cols) const {
		return nChooseK(n_cols, 2);
	}

    MatrixXd MIA::getPolynomialX(const MatrixXd& x) const {
        MatrixXd polyX(x);
        if ((polynomialType == PolynomialType::linear_cov)) {
            polyX.conservativeResize(NoChange, 4);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(2);
        }
        else if ((polynomialType == PolynomialType::quadratic)) {
            polyX.conservativeResize(NoChange, 6);
            polyX.col(2) = x.col(0).cwiseProduct(x.col(1));
            polyX.col(3) = x.col(0).cwiseProduct(x.col(0));
            polyX.col(4) = x.col(1).cwiseProduct(x.col(1));
            polyX.col(5) = x.col(2);
        }
        return polyX;
    }

    void MIA::transformDataForNextLevel(SplittedData& data, const VectorC& bestCombinations) {
        MatrixXd xTrainNew(data.xTrain.rows(), bestCombinations.size() + 1);
        MatrixXd xTestNew(data.xTest.rows(), bestCombinations.size() + 1);
        for (int i = 0; i < bestCombinations.size(); ++i) {
            auto comb = bestCombinations[i].combination();
            xTrainNew.col(i) = getPolynomialX(data.xTrain(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
            xTestNew.col(i) = getPolynomialX(data.xTest(Eigen::all, comb)) * bestCombinations[i].bestCoeffs();
        }
        xTrainNew.col(xTrainNew.cols() - 1) = VectorXd::Ones(xTrainNew.rows());
        xTestNew.col(xTestNew.cols() - 1) = VectorXd::Ones(xTestNew.rows());
        data.xTrain = xTrainNew;
        data.xTest = xTestNew;
    }

    void MIA::removeExtraCombinations() {
        std::vector<VectorC> realBestCombinations(bestCombinations.size());
        realBestCombinations[realBestCombinations.size() - 1] = VectorC(1, bestCombinations[level - 2][0]);
        for (int i = realBestCombinations.size() - 1; i > 0; --i) {
            std::set<uint16_t> usedCombinationsIndexes;
            for (int j = 0; j < realBestCombinations[i].size(); ++j) {
                auto comb = realBestCombinations[i][j].combination();
                for (int k = 0; k < comb.size() - 1; ++k)
                    usedCombinationsIndexes.insert(comb[k]);
            }
            for (auto it = usedCombinationsIndexes.begin(); it != usedCombinationsIndexes.end(); ++it)
                realBestCombinations[i - 1].push_back(bestCombinations[i - 1][*it]);
            for (int j = 0; j < realBestCombinations[i].size(); ++j) {
                auto comb = realBestCombinations[i][j].combination();
                for (int k = 0; k < comb.size() - 1; ++k)
                    comb[k] = std::distance(usedCombinationsIndexes.begin(), usedCombinationsIndexes.find(comb[k]));
                comb[comb.size() - 1] = usedCombinationsIndexes.size();
                realBestCombinations[i][j].setCombination(std::move(comb));
            }
        }
        bestCombinations = realBestCombinations;
    }

    bool MIA::preparations(SplittedData& data, VectorC& _bestCombinations) {
        bestCombinations.push_back(std::move(_bestCombinations));
        transformDataForNextLevel(data, bestCombinations[level - 1]);
        return true;
    }

    MatrixXd MIA::xDataForCombination(const MatrixXd& x, const VectorU16& comb) const {
        return getPolynomialX(x(Eigen::all, comb));
    }

    std::string MIA::getPolynomialPrefix(int levelIndex, int combIndex) const {
        return ((levelIndex < bestCombinations.size() - 1) ?
            "f" + std::to_string(levelIndex + 1) + "_" + std::to_string(combIndex + 1) : "y") + " =";
    }

    std::string MIA::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                           const VectorU16& bestColsIndexes) const {
        if (levelIndex == 0) {
            if (coeffIndex < 2)
                return "*x" + std::to_string(bestColsIndexes[coeffIndex] + 1);
            else if (coeffIndex == 2 && coeffsNumber > 3)
                return "*x" + std::to_string(bestColsIndexes[0] + 1) + 
                       "*x" + std::to_string(bestColsIndexes[1] + 1);
            else if (coeffIndex < 5 && coeffsNumber > 4)
                return "*x" + std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
        }
        else {
            if (coeffIndex < 2)
                return "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[coeffIndex] + 1);
            else if (coeffIndex == 2 && coeffsNumber > 3)
                return "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[0] + 1) +
                       "*f" + std::to_string(levelIndex) + "_" + std::to_string(bestColsIndexes[1] + 1);
            else if (coeffIndex < 5 && coeffsNumber > 4)
                return "*f" + std::to_string(levelIndex) + "_" + 
                       std::to_string(bestColsIndexes[coeffIndex - 3] + 1) + "^2";
        }
        return "";
    }

    GmdhModel& MIA::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
                        PolynomialType _polynomialType, double testSize, uint8_t pAverage, 
                        int threads, int verbose, double limit) {
        validateInputData(&testSize, &pAverage, &threads, &kBest);
        polynomialType = _polynomialType;
        return GmdhModel::gmdhFit(x, y, criterion, kBest, testSize, pAverage, threads, verbose, limit);
    }

    VectorXd MIA::predict(const MatrixXd& x) const {
        if (inputColsNumber != x.cols())
            throw GmdhException(GMDHPREDICTEXCEPTIONMSG); // TODO: maybe move to base class
        MatrixXd modifiedX(x.rows(), x.cols() + 1);
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        for (int i = 0; i < bestCombinations.size(); ++i) {
            MatrixXd xNew(x.rows(), bestCombinations[i].size() + 1);
            for (int j = 0; j < bestCombinations[i].size(); ++j) {
                auto comb = bestCombinations[i][j].combination();
                xNew.col(j) = getPolynomialX(modifiedX(Eigen::all, comb)) * bestCombinations[i][j].bestCoeffs();;
            }
            xNew.col(xNew.cols() - 1) = VectorXd::Ones(xNew.rows());
            modifiedX = xNew;
        }
        return modifiedX.col(0);
    }

    int MIA::save(const std::string& path) const {
        if (!boost::filesystem::is_regular_file(path))
#ifdef GMDH_MODULE
            throw GmdhException(GMDHOPENFILEEXCEPTIONMSG);
#else
            return 1;
#endif
        std::ofstream modelFile(path);
        if (!modelFile.is_open())
#ifdef GMDH_MODULE
            throw GmdhException(GMDHOPENFILEEXCEPTIONMSG);
#else
            return 1;
#endif
        else {
            modelFile << getModelName() << "\n" << inputColsNumber << "\n" << static_cast<int>(polynomialType) << "\n";
            for (int i = 0; i < bestCombinations.size(); ++i) {
                modelFile << "~\n";
                for (int j = 0; j < bestCombinations[i].size(); ++j)
                    modelFile << bestCombinations[i][j].getInfoForSaving();
            }
            modelFile.close();
        }
        return 0;
    }

    int MIA::load(const std::string& path) {
        auto getIntegerValue = [](std::stringstream& stream, auto& value) -> auto {
            if ((stream >> value).fail() && !stream.eof()) return 2;
            else if (stream.eof()) return 1;

            return 0;
        };

        if (!boost::filesystem::is_regular_file(path))
#ifdef GMDH_MODULE
            throw GmdhException(GMDHOPENFILEEXCEPTIONMSG);
#else
            return 1;
#endif

        std::ifstream modelFile(path);

        if (!modelFile.is_open())
#ifdef GMDH_MODULE
            throw GmdhException(GMDHOPENFILEEXCEPTIONMSG);
#else
            return 1;
#endif
        else {
            int newColsNumber;
            decltype(bestCombinations) newBestCombinations;

            std::string modelName;
            std::getline(modelFile, modelName);
            if (modelName != getModelName()) {
                modelFile.close();
#ifdef GMDH_MODULE
                throw GmdhException(GMDHLOADMODELNAMEEXCEPTIONMSG(modelName, getModelName()));
#else
                return 2;
#endif
            }
            else {
                auto errorCode = 0;
                (modelFile >> newColsNumber).get(); // TODO: add validation       ERROR: getIntegerValue(modelFile, inputColsNumber);
                if (errorCode == 2) {
                    modelFile.close();
#ifdef GMDH_MODULE
                    throw GmdhException(GMDHLOADMODELPARAMSEXCEPTIONMSG);
#else
                    return 3; // TODO: remove dublicate
#endif
                }

                int type;
                (modelFile >> type).get();
                polynomialType = static_cast<PolynomialType>(type);

                int currLevel = -1;

                while (modelFile.peek() != EOF) {

                    if (modelFile.peek() == '~') {
                        std::string buffer;
                        std::getline(modelFile, buffer);
                        ++currLevel;
                        newBestCombinations.push_back(VectorC());
                    }
                    else if (currLevel == -1) {
                        modelFile.close();
#ifdef GMDH_MODULE
                        throw GmdhException(GMDHLOADMODELPARAMSEXCEPTIONMSG);
#else
                        return 3;
#endif
                    }

                    std::string colsIndexesLine;
                    VectorU16 bestColsIndexes;
                    std::getline(modelFile, colsIndexesLine);
                    std::stringstream indexStream{ colsIndexesLine };
                    uint16_t index;
                    while (!(errorCode = getIntegerValue(indexStream, index)))
                        bestColsIndexes.push_back(index);

                    if (errorCode == 2) {
                        modelFile.close();
#ifdef GMDH_MODULE
                        throw GmdhException(GMDHLOADMODELPARAMSEXCEPTIONMSG);
#else
                        return 3; // TODO: remove dublicate
#endif
                    }
                    std::string coeffsLine;
                    std::vector<double> coeffs;
                    std::getline(modelFile, coeffsLine);
                    std::stringstream coeffsStream{ coeffsLine };
                    double coeff;
                    while (!(errorCode = getIntegerValue(coeffsStream, coeff)))
                        coeffs.push_back(coeff);

                    if (errorCode == 2) {
                        modelFile.close();
#ifdef GMDH_MODULE
                        throw GmdhException(GMDHLOADMODELPARAMSEXCEPTIONMSG);
#else
                        return 3; // TODO: remove dublicate
#endif
                    }
                    newBestCombinations[currLevel].push_back({ std::move(bestColsIndexes),
                                                            Map<VectorXd>(coeffs.data(), coeffs.size()) });
                }
            }
            modelFile.close();
            inputColsNumber = newColsNumber;
            bestCombinations = std::move(newBestCombinations);
        }
        return 0;
    }
}