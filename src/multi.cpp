//#include "pch.h"
#include "multi.h"

namespace GMDH {

    VectorVu16 MULTI::getCombinations(int n) const
    {
        VectorVu16 combs;
        if (level == 1)
            return nChooseK(n, level);

        for (auto comb : bestCombinations[0])
        {
            for (int i = 0; i < n; ++i)
            {
                VectorU16 temp(comb.combination());
                if (std::find(temp.begin(), temp.end(), i) == temp.end())
                {
                    temp.push_back(i);
                    std::sort(temp.begin(), temp.end());
                    if (std::find(combs.begin(), combs.end(), temp) == combs.end())
                        combs.push_back(temp);
                }
            }
        }
        return combs;
    }

    MULTI::MULTI()
    {
        bestCombinations.resize(1);
    }

    int MULTI::save(const std::string& path) const
    {
        std::ofstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1;
        else {
            modelFile << getModelName() << "\n";
            modelFile << inputColsNumber << "\n";
            modelFile << bestCombinations[0][0].getInfoForSaving();
            modelFile.close();
        }
        return 0;
    }

    int MULTI::load(const std::string& path)
    {
        inputColsNumber = 0;
        bestCombinations.clear();
        bestCombinations.resize(1);

        std::ifstream modelFile;
        modelFile.open(path);
        if (!modelFile.is_open())
            return -1;
        else {
            std::string modelName;
            modelFile >> modelName;
            if (modelName != getModelName())
                return -1;
            else {
                (modelFile >> inputColsNumber).get();

                std::string colsIndexesLine;
                VectorU16 bestColsIndexes;
                std::getline(modelFile, colsIndexesLine);
                std::stringstream indexStream(colsIndexesLine);
                uint16_t index;
                while (indexStream >> index)
                    bestColsIndexes.push_back(index);


                std::string coeffsLine;
                std::vector<double> coeffs;
                std::getline(modelFile, coeffsLine);
                std::stringstream coeffsStream(coeffsLine);
                double coeff;
                while (coeffsStream >> coeff)
                    coeffs.push_back(coeff);

                bestCombinations[0].push_back(Combination(std::move(bestColsIndexes), Map<VectorXd>(coeffs.data(), coeffs.size())));
            }
            modelFile.close();
        }
        return 0;
    }

    GMDH& MULTI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, double testSize, bool shuffle, int randomSeed, uint8_t p, int threads, int verbose)
    {
        return GMDH::fit(x, y, criterion, kBest, testSize, shuffle, randomSeed, p, threads, verbose);
    }

    double MULTI::predict(const RowVectorXd& x) const
    {
        return predict(MatrixXd(x))[0];
    }

    VectorXd MULTI::predict(const MatrixXd& x) const
    {
        MatrixXd modifiedX(x.rows(), x.cols() + 1);
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        return modifiedX(Eigen::all, bestCombinations[0][0].combination()) * bestCombinations[0][0].bestCoeffs();
    }

    std::string MULTI::getBestPolynomial() const
    {
        std::string polynomialStr = "y =";
        auto bestColsIndexes = bestCombinations[0][0].combination();
        auto bestCoeffs = bestCombinations[0][0].bestCoeffs();
        for (int i = 0; i < bestColsIndexes.size(); ++i) {
            if (bestCoeffs[i] > 0) {
                if (i > 0)
                    polynomialStr += " + ";
                else
                    polynomialStr += " ";
            }
            else
                polynomialStr += " - ";
            polynomialStr += std::to_string(abs(bestCoeffs[i]));
            if (i != bestColsIndexes.size() - 1)
                polynomialStr += "*x" + std::to_string(bestColsIndexes[i] + 1);
        }
        return polynomialStr;
    }
}
