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

class GMDH {

protected:
    int level;
    virtual std::string getModelName() const;
public:
    GMDH();
    virtual int save(const std::string& path) const = 0;
    virtual int load(const std::string& path) = 0;
    virtual GMDH& fit(MatrixXd x, VectorXd y, const Criterion& criterion, int threads = 1, int verbose = 0) = 0;
    virtual double predict(const RowVectorXd& x) const = 0;
    virtual VectorXd predict(const MatrixXd& x) const = 0;
    virtual std::string getBestPolynomial() const = 0;
};


//mat polynomailFeatures(const mat X, int max_degree);
std::pair<MatrixXd, VectorXd> convertToTimeSeries(VectorXd x, int lags);
SplittedData splitTimeSeries(MatrixXd x, VectorXd y, double testSize = 0.2);
SplittedData splitData(MatrixXd x, VectorXd y, double testSize = 0.2, bool shuffle = true, int randomSeed = 0);

}