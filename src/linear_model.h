#pragma once
#include "gmdh.h"

namespace GMDH {

    class GMDH_API LinearModel : public GmdhModel {
    protected:
        virtual void removeExtraCombinations() override;
        virtual bool preparations(SplittedData& data, VectorC& _bestCombinations) override;
        virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const override;

        std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
        std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                          const VectorU16& bestColsIndexes) const override;

        virtual VectorVu16 generateCombinations(int n_cols) const = 0;
    public:
        LinearModel();

        using GmdhModel::predict;
        VectorXd predict(const MatrixXd& x) const override;
    };
}