#include "multi.h"

namespace GMDH {

    VectorVu16 MULTI::generateCombinations(int n_cols) const {
        VectorVu16 combs;
        if (level == 1)
            return nChooseK(n_cols, level);

        for (auto comb : bestCombinations[0]) {
            for (auto i = 0; i < n_cols; ++i) {
                auto temp{ comb.combination() };
                if (std::find(std::begin(temp), std::end(temp), i) == std::end(temp)) {
                    temp.push_back(i);
                    std::sort(std::begin(temp), std::end(temp));
                    if (std::find(std::begin(combs), std::end(combs), temp) == std::end(combs))
                        combs.push_back(std::move(temp));
                }
            }
        }
        return combs;
    }

    std::string MULTI::getPolynomialPrefix(int levelIndex, int combIndex) const {
        return "y =";
    }

    std::string MULTI::getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const {
        return ((coeffIndex != coeffsNumber - 1) ? "*x" + std::to_string(bestColsIndexes[coeffIndex] + 1) : "");
    }

    MULTI::MULTI() {
        bestCombinations.resize(1);
    }

    int MULTI::save(const std::string& path) const {
        std::ofstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1; // TODO: throw exception for Python???
        else {
            modelFile << getModelName() << "\n"
            << inputColsNumber << "\n"
            << bestCombinations[0][0].getInfoForSaving();
            modelFile.close();
        }
        return 0;
    }

    int MULTI::load(const std::string& path) {
        inputColsNumber = 0;
        bestCombinations.clear();
        bestCombinations.resize(1);

        std::ifstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1; // TODO: throw exception for Python???
        else {
            std::string modelName;
            modelFile >> modelName;
            if (modelName != getModelName())
                return -2; // TODO: throw exception for Python???
            else {
                (modelFile >> inputColsNumber).get();

                std::string colsIndexesLine;
                VectorU16 bestColsIndexes;
                std::getline(modelFile, colsIndexesLine);
                std::stringstream indexStream{ colsIndexesLine };
                uint16_t index;
                while (indexStream >> index)
                    bestColsIndexes.push_back(index);


                std::string coeffsLine;
                std::vector<double> coeffs;
                std::getline(modelFile, coeffsLine);
                std::stringstream coeffsStream{ coeffsLine };
                double coeff;
                while (coeffsStream >> coeff)
                    coeffs.push_back(coeff);

                bestCombinations[0].push_back({ std::move(bestColsIndexes), Map<VectorXd>(coeffs.data(), coeffs.size()) });
            }
            modelFile.close();
        }
        return 0;
    }

    GmdhModel& MULTI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int _kBest, double testSize, // TODO: whaaat? why not kBest without '_'
                     bool shuffle, int randomSeed, uint8_t pAverage, int threads, int verbose, double limit) {
        validateInputData(&testSize, &pAverage, &threads, &_kBest);
        return GmdhModel::fit(x, y, criterion, _kBest, testSize, shuffle, randomSeed, pAverage, threads, verbose, limit);
    }

    double MULTI::predict(const RowVectorXd& x) const {
        return predict(MatrixXd(x))[0];
    }

    VectorXd MULTI::predict(const MatrixXd& x) const {
        if (inputColsNumber != x.cols())
            throw GmdhException(GMDHPREDICTEXCEPTIONMSG);
        MatrixXd modifiedX{ x.rows(), x.cols() + 1 };
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        return modifiedX(Eigen::all, bestCombinations[0][0].combination()) * bestCombinations[0][0].bestCoeffs();
    }
}
