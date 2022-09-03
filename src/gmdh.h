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

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/block_progress_bar.hpp>

#include "gmdh_lib.h"
#include "common.h"
#include "gmdh_objects.h"

namespace GMDH {    

class GMDH_API GMDH {
    int verifyInputData(uint8_t& pAverage, int& threads) const; // TODO: add verify testSize
    //int calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > beginTasksVec, 
    //const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > endTasksVec) const;
protected:

    int level;
    int inputColsNumber;
    std::vector<VectorC> bestCombinations;

    std::string getModelName() const;
    VectorVu16 nChooseK(int n, int k) const;
    virtual VectorVu16 generateCombinations(int n_cols) const = 0;
    virtual VectorC getBestCombinations(VectorC& combinations, int k) const;
    double getMeanCriterionValue(const VectorC& sortedCombinations, int k) const;

    virtual void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
                                       IterC endCoeffsVec, std::atomic<int>* leftTasks, bool verbose) const;

    virtual bool nextLevelCondition(double& lastLevelEvaluation, int kBest, uint8_t pAverage, VectorC& combinations,
                                    const Criterion& criterion, SplittedData& data, double limit);
   
    GMDH& fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
              double testSize = 0.5, bool shuffle = false, int randomSeed = 0, uint8_t pAverage = 1, 
              int threads = 1, int verbose = 0, double limit = 0);

    virtual std::string getPolynomialPrefix(int levelIndex, int combIndex) const = 0;
    virtual std::string getPolynomialCoeff(double coeff, int coeffIndex) const;
    virtual std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, const VectorU16& bestColsIndexes) const = 0;

public:
    GMDH() : level(1) {}
    virtual int save(const std::string& path) const = 0;
    virtual int load(const std::string& path) = 0;

    virtual double predict(const RowVectorXd& x) const = 0;
    virtual VectorXd predict(const MatrixXd& x) const = 0;
    std::string getBestPolynomial() const;
};

PairMVXd GMDH_API timeSeriesTransformation(VectorXd x, int lags);
SplittedData GMDH_API splitData(const MatrixXd& x, const VectorXd& y, double testSize = 0.2, 
                                bool shuffle = false, int randomSeed = 0);
}

#include "criterion.h"