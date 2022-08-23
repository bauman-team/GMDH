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

#include <Eigen/Dense>

#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/future.hpp>
#include <boost/type_index.hpp>

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/block_progress_bar.hpp>

#include "gmdh_lib.h"

#ifdef __GNUC__
    #define likely(expr)    (__builtin_expect(!!(expr), 1))
    #define unlikely(expr)  (__builtin_expect(!!(expr), 0))
#else
    #define likely(expr)    expr
    #define unlikely(expr)  expr
#endif

namespace GMDH {

using namespace Eigen;

using VectorU16 = std::vector<uint16_t>;
using IterU16 = VectorU16::iterator;
using VectorVu16 = std::vector<VectorU16>;
using PairDVXd = std::pair<double, VectorXd>;
using PairMVXd = std::pair<MatrixXd, VectorXd>;
using VectorI = std::vector<int>;

class Criterion;

struct GMDH_API SplittedData {
    MatrixXd xTrain;
    MatrixXd xTest;
    VectorXd yTrain;
    VectorXd yTest;
};

class GMDH_API Combination { // TODO: move to separate file
    VectorU16 _combination;
    VectorXd _bestCoeffs;
    double _evaluation;
public:
    Combination() {}
    Combination(VectorU16 comb) : _combination(comb) {} 
    Combination(VectorU16&& comb, VectorXd&& coeffs) : _combination(std::move(comb)), _bestCoeffs(std::move(coeffs)) {} 
    const VectorU16& combination() const { return _combination; }
    const VectorXd& bestCoeffs() const { return _bestCoeffs; }
    double evaluation() const { return _evaluation; }

    void setCombination(VectorU16&& combination) { _combination = std::move(combination); }
    void setBestCoeffs(VectorXd&& bestCoeffs) { _bestCoeffs = std::move(bestCoeffs);}
    void setEvaluation(double evaluation) { _evaluation = evaluation; }

    bool operator<(const Combination& comb) { return _evaluation < comb._evaluation; }

};

using VectorC = std::vector<Combination>;
using IterC = VectorC::iterator;
using cIterC = VectorC::const_iterator;
    

class GMDH_API GMDH {
    void polinomialsEvaluation(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, 
        IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, bool verbose) const;
    virtual bool nextLevelCondition(double &lastLevelEvaluation, uint8_t p, VectorC& combinations);
    //int calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > beginTasksVec, 
    //const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > endTasksVec) const;
protected:

    int level;
    int inputColsNumber; // TODO: maybe delete???
    VectorC bestCombinations;

    std::string getModelName() const;
    virtual VectorVu16 getCombinations(int n, int k) const = 0;

public:
    GMDH() : level(1) { }
    virtual int save(const std::string& path) const = 0;
    virtual int load(const std::string& path) = 0;
    GMDH& fit(MatrixXd x, VectorXd y, const Criterion& criterion, uint8_t p = 1, int threads = 1, int verbose = 0);
    virtual double predict(const RowVectorXd& x) const = 0;
    virtual VectorXd predict(const MatrixXd& x) const = 0;
    virtual std::string getBestPolynomial() const = 0;
};


//mat polynomailFeatures(const mat X, int max_degree);
PairMVXd GMDH_API convertToTimeSeries(VectorXd x, int lags);
SplittedData GMDH_API splitTimeSeries(MatrixXd x, VectorXd y, double testSize = 0.2);
SplittedData GMDH_API splitData(MatrixXd x, VectorXd y, double testSize = 0.2, bool shuffle = true, int randomSeed = 0);

}

#include "criterion.h"