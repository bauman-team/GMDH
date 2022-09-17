#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY

#include <vector>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <atomic>
#include <map>
#include <set>

#include <Eigen/Dense>

#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/type_index.hpp>
#include <boost/chrono.hpp>
#include <boost/algorithm/string.hpp>

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/block_progress_bar.hpp>

#include "gmdh_lib.h"
#include "common.h"
#include "gmdh_objects.h"

namespace GMDH {    

class GMDH_API GmdhModel { 
    //int calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > beginTasksVec, 
    //const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > endTasksVec) const; 
protected:

    int level;
    int inputColsNumber;
    double lastLevelEvaluation;
    double currentLevelEvaluation;
    std::vector<VectorC> bestCombinations;

    std::string getModelName() const;
    VectorVu16 nChooseK(int n, int k) const;
    double getMeanCriterionValue(const VectorC& sortedCombinations, int k) const;
    std::string getPolynomialCoeff(double coeff, int coeffIndex) const;
    void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
                               IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const;
    bool nextLevelCondition(int kBest, uint8_t pAverage, VectorC& combinations,
                            const Criterion& criterion, SplittedData& data, double limit);
    GmdhModel& gmdhFit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest,
                   double testSize, uint8_t pAverage, int threads, int verbose, double limit);

    virtual VectorVu16 generateCombinations(int n_cols) const = 0;
    virtual void removeExtraCombinations() = 0;
    virtual bool preparations(SplittedData& data, VectorC& _bestCombinations) = 0;
    virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const = 0;
    virtual std::string getPolynomialPrefix(int levelIndex, int combIndex) const = 0;
    virtual std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                              const VectorU16& bestColsIndexes) const = 0;

public:
    GmdhModel() : level(1), lastLevelEvaluation(0) {}

    int save(const std::string& path) const;
    int load(const std::string& path);

    VectorXd predict(const RowVectorXd& x, int lags) const;
    virtual VectorXd predict(const MatrixXd& x) const = 0;
    std::string getBestPolynomial() const;
};

int GMDH_API validateInputData(double *testSize, uint8_t *pAverage = nullptr, 
                               int *threads = nullptr, int *kBest = nullptr);
PairMVXd GMDH_API timeSeriesTransformation(VectorXd x, int lags);
SplittedData GMDH_API splitData(const MatrixXd& x, const VectorXd& y, double testSize = 0.2, 
                                bool shuffle = false, int randomSeed = 0);
}

#include "criterion.h"