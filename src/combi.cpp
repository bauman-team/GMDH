//#include "pch.h"
#include "combi.h"

namespace GMDH {

int COMBI::save(const std::string& path) const
{
    std::ofstream modelFile;
    modelFile.open(path);
    if (!modelFile.is_open())
        return -1;
    else {
        modelFile << getModelName() << "\n";
        modelFile << inputColsNumber << "\n";
        for (auto i : bestCombinations[0].combination()) modelFile << i << ' ';
        modelFile << "\n";
        modelFile.precision(10); // TODOL maybe change precision
        for (auto i : bestCombinations[0].bestCoeffs()) modelFile << i << ' ';
        modelFile << "\n";
        modelFile.close();
    }
    return 0;
}

int COMBI::load(const std::string& path)
{
    inputColsNumber = 0;
    bestCombinations.clear();
    VectorU16 bestColsIndexes;

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

            bestCombinations.push_back(Combination(std::move(bestColsIndexes), Map<VectorXd>(coeffs.data(), coeffs.size())));
        }
        modelFile.close();
    }
    return 0;
}

GMDH& COMBI::fit(const MatrixXd& x, const VectorXd& y, Criterion& criterion, double testSize, bool shuffle, int randomSeed, uint8_t p, int threads, int verbose)
{
    kBest = p;
    return GMDH::fit(x, y, criterion, testSize, shuffle, randomSeed, p, threads, verbose);
}

double COMBI::predict(const RowVectorXd& x) const
{
    return predict(MatrixXd(x))[0];
}

VectorXd COMBI::predict(const MatrixXd& x) const
{
    MatrixXd modifiedX(x.rows(), x.cols() + 1);
    modifiedX.col(x.cols()).setOnes();
    modifiedX.leftCols(x.cols()) = x;
    return modifiedX(Eigen::all, bestCombinations[0].combination()) * bestCombinations[0].bestCoeffs();
}

VectorVu16 COMBI::getCombinations(int n, int k) const
{
    struct c_unique {
        uint16_t current;
        c_unique() { current = -1; }
        uint16_t operator()() { return ++current; }
    } UniqueNumber;

    VectorVu16 combs;
    VectorU16 comb(k);
    IterU16 first = comb.begin(), last = comb.end();

    std::generate(first, last, UniqueNumber);
    combs.push_back(comb);

    while ((*first) != n - k) {
        IterU16 mt = last;
        while (*(--mt) == n - (last - mt));
        (*mt)++;
        while (++mt != last) *mt = *(mt - 1) + 1;
        combs.push_back(comb);
    }

    for (int i = 0; i < combs.size(); ++i)
        combs[i].push_back(n);
    return combs;
}

std::string COMBI::getBestPolynomial() const
{
    std::string polynomialStr = "y =";
    auto bestColsIndexes = bestCombinations[0].combination();
    auto bestCoeffs = bestCombinations[0].bestCoeffs();
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