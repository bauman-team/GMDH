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

#include <Dense>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/future.hpp>




namespace GMDH {

using namespace Eigen;


struct splitted_data {
    MatrixXd x_train;
    MatrixXd x_test;
    VectorXd y_train;
    VectorXd y_test;
};

class Criterion {

protected:
    VectorXd internalCriterion(MatrixXd x_train, VectorXd y_train) const;

public:
    virtual std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const = 0;
};

class RegularityCriterionTS : public Criterion
{
protected:
    double test_size;

    std::pair<double, VectorXd> getCriterionValue(splitted_data data) const;
public:
    RegularityCriterionTS(double _test_size = 0.33);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};

class RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int random_seed;

public:
    RegularityCriterion(double _test_size = 0.33, bool _shuffle = true, int _random_seed = 0);
    std::pair<double, VectorXd> calculate(const MatrixXd& x, const VectorXd& y) const override;
};


class GMDH {

protected:
    int level;
    std::string model_name;

public:
    GMDH();
    virtual int save(std::string path) const = 0;
    virtual int load(const std::string& path) = 0;
    virtual GMDH& fit(MatrixXd x, VectorXd y, const Criterion& criterion) = 0;
    virtual double predict(RowVectorXd x) const = 0;
    virtual VectorXd predict(MatrixXd x) const = 0;
    virtual std::string getBestPolymon() const = 0;
};

class COMBI : public GMDH { // TODO: split into separate files

    std::vector<int> best_cols_index;
    VectorXd best_coeffs;
    int input_cols_number;

    std::vector<std::vector<bool>> getCombinations(int n, int k) const;
    std::vector<int> polinomToIndexes(std::vector<bool> polinom) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    COMBI();
    int save(std::string path) const override;
    int load(const std::string& path) override;
    double predict(RowVectorXd x) const override;
    VectorXd predict(MatrixXd x) const override;
    COMBI& fit(MatrixXd x, VectorXd y, const Criterion& criterion) override;
    std::string getBestPolymon() const override;
};

//mat polynomailFeatures(const mat X, int max_degree);
std::pair<MatrixXd, VectorXd> convertToTimeSeries(VectorXd x, int lags);
splitted_data splitTsData(MatrixXd x, VectorXd y, double validate_size = 0.2);
splitted_data splitData(MatrixXd x, VectorXd y, double validate_size = 0.2, bool shuffle = true, int _random_seed = 0);

}