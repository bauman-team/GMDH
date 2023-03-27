#pragma once
#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#define EIGEN_MPL2_ONLY

#include <vector>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
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

/// @brief Namespace containing the functionality of Group Method of Data Handling
namespace GMDH {    

/**
 * @brief Divide the input data into 2 parts
 * 
 * @param x Matrix of input data containing predictive variables
 * @param y Vector of the taget values for the corresponding x data
 * @param testSize Fraction of the input data that should be placed into the second part
 * @param shuffle True if data should be shuffled before splitting into 2 parts, otherwise false
 * @param randomSeed Seed number for the random generator to get the same division every time 
 * @throw std::invalid_argument
 * @return SplittedData object containing 4 elements of data: train x, train y, test x, test y
 */
SplittedData GMDH_API splitData(const MatrixXd& x, const VectorXd& y, double testSize = 0.2,
    bool shuffle = false, int randomSeed = 0);

/// @brief Class implementing the general logic of GMDH algorithms
class GMDH_API GmdhModel { 
    //int calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > beginTasksVec, 
    //const std::vector<std::shared_ptr<std::vector<Combination>::iterator> > endTasksVec) const; 
protected:

    int level; //!< Current number of the algorithm training level
    int inputColsNumber; //!< The number of predictive variables in the original data
    double lastLevelEvaluation; //!< The external criterion value of the previous training level
    double currentLevelEvaluation; //!< The external criterion value of the current training level
    std::vector<VectorC> bestCombinations; //!< Storage for the best models of previous levels

    /**
     * @brief Get full class name
     * 
     * @return String containing the name of the model class
     */
    std::string getModelName() const;

    /**
     * @brief Find all combinations of k elements from n
     * 
     * @param n Number of all elements
     * @param k Number of required elements
     * @return Vector of all combinations of k elements from n 
     */
    VectorVu16 nChooseK(int n, int k) const;

    /**
     * @brief Get the mean value of extrnal criterion of the k best models
     * 
     * @param sortedCombinations Sorted vector of current level models
     * @param k The numebr of the best models
     * @return Calculated mean value of extrnal criterion of the k best models
     */
    double getMeanCriterionValue(const VectorC& sortedCombinations, int k) const;

    /**
     * @brief Get the sign of the polynomial variable coefficient
     * 
     * @param coeff Selected coefficient
     * @param isFirstCoeff True if the selected coefficient will be the first in the polynomial representation, otherwise false 
     * @return String containing the sign of the coefficient
     */
    std::string getPolynomialCoeffSign(double coeff, bool isFirstCoeff) const;

    /**
     * @brief Get the rounded value of the polynomial variable coefficient without sign
     * 
     * @param coeff Selected coefficient
     * @param isLastCoeff True if the selected coefficient will be the last one in the polynomial representation, otherwise false 
     * @return String containing the rounded value of the coefficient without sign
     */
    std::string getPolynomialCoeffValue(double coeff, bool isLastCoeff) const;

    /**
     * @brief Train given subset of models and calculate external criterion for them
     * 
     * @param data Data used for training and evaulating models
     * @param criterion Selected external criterion
     * @param beginCoeffsVec Iterator indicating the beginning of a subset of models
     * @param endCoeffsVec Iterator indicating the end of a subset of models
     * @param leftTasks The number of remaining untrained models at the entire level
     * @param verbose 1 if the printing detailed infomation about training process is needed, otherwise 0
     */
    void polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, IterC beginCoeffsVec, 
                               IterC endCoeffsVec, std::atomic<int>* leftTasks, int verbose) const;

    /**
    * @brief Determine the need to continue training and prepare the algorithm for the next level
    * 
    * @param kBest The number of best models based of which new models of the next level will be constructed
    * @param pAverage The number of best models based of which the external criterion for each level will be calculated
    * @param combinations Trained models of the current level
    * @param criterion Selected external criterion
    * @param data Data used for training and evaulating models
    * @param limit The minimum value by which the external criterion should be improved in order to continue training
    * @return True if the algorithm needs to continue training, otherwise fasle 
    */
    bool nextLevelCondition(int kBest, int pAverage, VectorC& combinations,
                            const Criterion& criterion, SplittedData& data, double limit);

    /**
     * @brief Fit the algorithm to find the best solution
     * 
     * @param x Matrix of input data containing predictive variables
     * @param y Vector of the taget values for the corresponding x data
     * @param criterion Selected external criterion
     * @param kBest The number of best models based of which new models of the next level will be constructed
     * @param testSize Fraction of the input data that should be used to evaluate models at each level
     * @param pAverage The number of best models based of which the external criterion for each level will be calculated
     * @param threads The number of threads used for calculations. Set -1 to use max possible threads 
     * @param verbose 1 if the printing detailed infomation about training process is needed, otherwise 0
     * @param limit The minimum value by which the external criterion should be improved in order to continue training
     * @throw std::invalid_argument
     * @warning If the threads or verbose value is incorrect an exception won't be thrown. 
     * Insted, the incorrect value will be replaced with the default value and a corresponding warning will be displayed
     * @return A reference to the algorithm object for which the training was performed
     */
    GmdhModel& gmdhFit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
                       double testSize, int pAverage, int threads, int verbose, double limit);

    /**
     * @brief Get new model structures for the new level of training
     * 
     * @param n_cols The number of existing predictive variables at the current training level
     * @return Vector of new model structures
     */
    virtual VectorVu16 generateCombinations(int n_cols) const = 0;

    /// @brief Removed the saved models that are no longer needed
    virtual void removeExtraCombinations() = 0;

    /**
     * @brief Prepare data for the next training level
     * 
     * @param data Data used for training and evaulating models at the current level
     * @param _bestCombinations Vector of the k best models of the current level
     * @return True if the training process can be continued, otherwise false 
     */
    virtual bool preparations(SplittedData& data, VectorC&& _bestCombinations) = 0;

    /**
     * @brief Get the data constructed according to the model structure from the original data
     * 
     * @param x Training data at the current level
     * @param comb Vector containing the indexes of the x matrix columns that should be used in the model
     * @return Constructed data 
     */
    virtual MatrixXd xDataForCombination(const MatrixXd& x, const VectorU16& comb) const = 0;

    /**
     * @brief Get the designation of polynomial equation
     * 
     * @param levelIndex The number of the level counting from 0
     * @param combIndex The number of polynomial in the level counting from 0
     * @return The designation of polynomial equation
     */
    virtual std::string getPolynomialPrefix(int levelIndex, int combIndex) const = 0;

    /**
     * @brief Get the string representation of the polynomial variable
     * 
     * @param levelIndex The number of the level counting from 0
     * @param coeffIndex The number of the coefficient related to the selected variable in the polynomial counting from 0
     * @param coeffsNumber The number of coefficients in the polynomial
     * @param bestColsIndexes Indexes of the data columns used to construct polynomial of the model
     * @return The string representation of the polynomial variable
     */
    virtual std::string getPolynomialVariable(int levelIndex, int coeffIndex, int coeffsNumber, 
                                              const VectorU16& bestColsIndexes) const = 0;

    /**
     * @brief Transform model data to JSON format for further saving
     * 
     * @return JSON value of model data
     */
    virtual boost::json::value toJSON() const;

    /**
     * @brief Set up model from JSON format model data
     *
     * @param jsonModel Model data in JSON format
     * @return Method exit status
     */
    virtual int fromJSON(boost::json::value jsonModel);

    /**
     * @brief Divide the input data into 2 parts without shuffling
     * 
     * @param x Matrix of input data containing predictive variables
     * @param y Vector of the taget values for the corresponding x data
     * @param testSize Fraction of the input data that should be placed into the second part
     * @param addOnesCol True if it is needed to add a column of ones to the x data, otherwise false
     * @return SplittedData object containing 4 elements of data: train x, train y, test x, test y 
     */
    static SplittedData internalSplitData(const MatrixXd& x, const VectorXd& y, double testSize, bool addOnesCol = false);

    friend SplittedData splitData(const MatrixXd& x, const VectorXd& y, double testSize,
                                           bool shuffle, int randomSeed);

    /**
     * @brief Compare the number of required and actual columns of the input matrix
     * 
     * @param x Given matrix of input data
     * @throw std::invalid_argument if the number of actual columns of the input matrix isn't equal to the required columns number
     */
    void checkMatrixColsNumber(const MatrixXd& x) const;
public:
    /// @brief Construct a new Gmdh Model object
    GmdhModel() : level(1), lastLevelEvaluation(0) {}

    /**
     * @brief Save model data into regular file
     *
     * @param path Path to regular file
     * @throw GMDH::FileException
     * @return Method exit status
     */
    int save(const std::string& path) const;

    /**
     * @brief Load model data from regular file
     *
     * @param path Path to regular file
     * @throw GMDH::FileException
     * @warning If the opening file has a large size then program falls without exceptions
     * @return Method exit status
     */
    int load(const std::string& path);

    /**
     * @brief Get long-term forecast for the time series
     * 
     * @param x One row of the test time series data
     * @param lags The number of lags (steps) to make a forecast for
     * @throw std::invalid_argument
     * @return Vector containing long-term forecast
     */
    VectorXd predict(const RowVectorXd& x, int lags) const;

    /**
     * @brief Get predictions for the input data
     * 
     * @param x Test data of the regression task or one-step time series forecast
     * @throw std::invalid_argument
     * @return Vector containing prediction values
     */
    virtual VectorXd predict(const MatrixXd& x) const = 0;

    /**
     * @brief Get the String representation of the best polynomial
     * 
     * @return String representation of the best polynomial
     */
    std::string getBestPolynomial() const;

    /// @brief Destroy the GmdhModel object
    virtual ~GmdhModel() {};
};

/**
 * @brief Validate input parameters values
 *
 * @param testSize Fraction of the input data that should be placed into the second part
 * @param pAverage The number of best models based of which the external criterion for each level will be calculated
 * @param threads The number of threads used for calculations. Set -1 to use max possible threads
 * @param verbose 1 if the printing detailed infomation about training process is needed, otherwise 0
 * @param limit The minimum value by which the external criterion should be improved in order to continue training
 * @param kBest The number of best models based of which new models of the next level will be constructed
 * @throw std::invalid_argument if testSize, pAverage, limit or kBest value is incorrect
 * @warning If the threads or verbose value is incorrect an exception won't be thrown. 
 * Insted, the incorrect value will be replaced with the default value and a corresponding warning will be displayed
 * @return Method exit status
 */
int GMDH_API validateInputData(double* testSize, int* pAverage = nullptr, int* threads = nullptr,
                               int* verbose = nullptr, double* limit = nullptr, int* kBest = nullptr);

/**
 * @brief Choose the correct variable name for the warning or exception text according to the GMDH_MODULE define
 * 
 * @param pyName correct variable name for the bound Python module
 * @param cppName correct veriable name for C++ library
 * @return pyName if GMDH_MODULE define else cppName
 */
std::string&& getVariableName(std::string&& pyName, std::string&& cppName);

/**
 * @brief Convert the time series vector to the 2D matrix format required to work with GMDH algorithms
 * 
 * @param timeSeries Vector of time series data
 * @param lags The lags (length) of subsets of time series into which the original time series should be divided
 * @throw std::invalid_argument
 * @return Transformed time series data
 */
PairMVXd GMDH_API timeSeriesTransformation(const VectorXd& timeSeries, int lags);
}