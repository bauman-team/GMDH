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
#include <boost/filesystem.hpp>
#include <boost/json.hpp>

#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/block_progress_bar.hpp>

#include "gmdh_lib.h"
#include "common.h"
#include "gmdh_objects.h"
#include "criterion.h"

namespace GMDH {    

SplittedData GMDH_API splitData(const MatrixXd& x, const VectorXd& y, double testSize = 0.2,
    bool shuffle = false, int randomSeed = 0);

class GMDH_API GmdhModel { 
    //int calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > beginTasksVec, 
    //const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > endTasksVec) const; 
protected:

    int level;
    int inputColsNumber;
    double lastLevelEvaluation;
    double currentLevelEvaluation;
    std::vector<VectorC> bestCombinations;


    /**
     * @brief Get full class name
     * 
     * @return String of class name model
     */
    std::string getModelName() const;

    VectorVu16 nChooseK(int n, int k) const;
    double getMeanCriterionValue(const VectorC& sortedCombinations, int k) const;
    std::string getPolynomialCoeff(double coeff, int coeffIndex) const;
    void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
                               IterC endCoeffsVec, std::atomic<int>* leftTasks, int verbose) const;
    bool nextLevelCondition(int kBest, int pAverage, VectorC& combinations,
                            const Criterion& criterion, SplittedData& data, double limit);
    GmdhModel& gmdhFit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
                       double testSize, int pAverage, int threads, int verbose, double limit);

    virtual VectorVu16 generateCombinations(int n_cols) const = 0;
    virtual void removeExtraCombinations() = 0;
    virtual bool preparations(SplittedData& data, VectorC&& _bestCombinations) = 0;
    virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const = 0;
    virtual std::string getPolynomialPrefix(int levelIndex, int combIndex) const = 0;
    virtual std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                              const VectorU16& bestColsIndexes) const = 0;

    /**
     * @brief Transform model data to JSON format for saving
     * 
     * @return JSON value of model data
     */
    virtual boost::json::value toJSON() const;

    /**
     * @brief Set up model from JSON format model data
     *
     * @param jsonModel model data in JSON format
     * 
     * @return Method exit status
     */
    virtual int fromJSON(boost::json::value jsonModel);

    static SplittedData internalSplitData(const MatrixXd& x, const VectorXd& y, double testSize, bool addOnesCol = false);

    friend SplittedData splitData(const MatrixXd& x, const VectorXd& y, double testSize,
                                           bool shuffle, int randomSeed);

    void checkMatrixColsNumber(const MatrixXd& x) const;
public:
    GmdhModel() : level(1), lastLevelEvaluation(0) {}

    /**
     * @brief Save model data into regular file
     *
     * @param path path to regular file
     * 
     * @return Method exit status
     */
    int save(const std::string& path) const;

    /**
     * @brief Load model data from regular file
     *
     * @param path path to regular file
     * 
     * @warning if opening file has a large size then program falls without exceptions
     * 
     * @throw if the opening file JSON structure is broken can throw exceptions as std::invalid_argument or std::out_of_range
     * 
     * @return Method exit status
     */
    int load(const std::string& path);

    VectorXd predict(const RowVectorXd& x, int lags) const;
    virtual VectorXd predict(const MatrixXd& x) const = 0;
    std::string getBestPolynomial() const;
};

/**
 * @brief Validate input params values and correct erroneous
 *
 * @param testSize value size of test selection
 * @param pAverage value of ...
 * @param threads value of using threads for fit
 * @param kBest value of ...
 * 
 * Exit status stores info about erroneous values in bit format
 * if testSize is erroneous then status = 0b1
 * if pAverage is erroneous then status = 0b10
 * if threads is erroneous then status = 0b100
 * ...
 * 
 * @return Method exit status
 */
int GMDH_API validateInputData(double* testSize, int* pAverage = nullptr, int* threads = nullptr,
                               int* verbose = nullptr, double* limit = nullptr, int* kBest = nullptr);
std::string&& getVariableName(std::string&& pyName, std::string&& cppName);
PairMVXd GMDH_API timeSeriesTransformation(const VectorXd& timeSeries, int lags);
}