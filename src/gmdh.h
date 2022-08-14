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

namespace GMDH {

using namespace Eigen;

class Criterion;

struct SplittedData {
    MatrixXd xTrain;
    MatrixXd xTest;
    VectorXd yTrain;
    VectorXd yTest;
};

class Combination {
    std::vector<uint16_t> _combination;
    VectorXd _bestCoeffs;
    double _evaluation;
public:
    Combination() {}
    Combination(std::vector<uint16_t> comb, VectorXd coeffs) : _combination(comb), _bestCoeffs(coeffs) {} // TODO: maybe std::move
    const std::vector<uint16_t>& combination() const { return _combination; }
    const VectorXd& bestCoeffs() const { return _bestCoeffs; }
    double evaluation() const { return _evaluation; }

    void setCombination(std::vector<uint16_t>&& combination) { _combination = std::forward<decltype(combination)>(combination); }
    void setBestCoeffs(VectorXd&& bestCoeffs) { _bestCoeffs = std::forward<decltype(bestCoeffs)>(bestCoeffs);}
    void setEvaluation(double evaluation) { _evaluation = evaluation; }

};

class GMDH {
    void polinomialsEvaluation(const MatrixXd& x, const VectorXd& y, 
    const Criterion& criterion, std::vector<Combination>::iterator beginCoeffsVec, std::vector<Combination>::iterator endCoeffsVec) const;
    virtual std::vector<std::vector<uint16_t>> getCombinations(int n, int k) const = 0;
    virtual bool nextLevelCondition(double &lastLevelEvaluation, uint8_t p, std::vector<Combination>& combinations);

    //virtual std::vector<int> polynomialToIndexes(const std::vector<bool>& polynomial) = 0;
protected:

    int level;
    int inputColsNumber; // TODO: maybe delete???
    std::vector<Combination> bestCombinations; // TODO: maybe multimap
    virtual std::string getModelName() const; // TODO: virtual delete

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
std::pair<MatrixXd, VectorXd> convertToTimeSeries(VectorXd x, int lags);
SplittedData splitTimeSeries(MatrixXd x, VectorXd y, double testSize = 0.2);
SplittedData splitData(MatrixXd x, VectorXd y, double testSize = 0.2, bool shuffle = true, int randomSeed = 0);

}

#include "criterion.h"