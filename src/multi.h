#include "gmdh.h"

namespace GMDH {

    class GMDH_API MULTI : public GmdhModel {
    protected:
        virtual void removeExtraCombinations() override;
        virtual bool preparations(SplittedData& data, VectorC& _bestCombinations) override;
        virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const override;

        VectorVu16 generateCombinations(int n_cols) const override;
        std::string getPolynomialPrefix(int levelIndex, int combIndex) const override;
        std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                          const VectorU16& bestColsIndexes) const override;
    public:
        MULTI();

        GmdhModel& fit(const MatrixXd& x, const VectorXd& y, 
                       const Criterion& criterion = Criterion(CriterionType::regularity), int kBest = 3,
                       double testSize = 0.5, uint8_t pAverage = 1, int threads = 1, int verbose = 0, double limit = 0);

        using GmdhModel::predict;
        VectorXd predict(const MatrixXd& x) const override;
    };
}