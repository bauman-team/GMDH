#define BOOST_THREAD_PROVIDES_FUTURE_WHEN_ALL_WHEN_ANY
#include <vector>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/future.hpp>




namespace GMDH {

using namespace arma;

class Criterion {

protected:
    vec internalCriterion(mat x_train, vec y_train) const;

public:
    virtual std::pair<double, vec> calculate(const mat& x, const vec& y) const = 0;
};

class RegularityCriterionTS : public Criterion
{
protected:
    double test_size;

    std::pair<double, vec> getCriterionValue(mat x_train, vec y_train, mat x_test, vec y_test) const;
public:
    RegularityCriterionTS(double _test_size = 0.33);
    std::pair<double, vec> calculate(const mat& x, const vec& y) const override;
};

class RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int random_seed;

public:
    RegularityCriterion(double _test_size = 0.33, bool _shuffle = true, int _random_seed = 0);
    std::pair<double, vec> calculate(const mat& x, const vec& y) const override;
};


class GMDH {

protected:
    int level;
    std::string model_name;

public:
    GMDH();
    virtual int save(std::string path) const = 0;
    virtual int load(const std::string& path) = 0;
    virtual GMDH& fit(mat x, vec y, const Criterion& criterion) = 0;
    virtual double predict(rowvec x) const = 0;
    virtual vec predict(mat x) const = 0;
};

class COMBI : public GMDH { // TODO: split into separate files

    uvec best_cols_index;
    vec best_coeffs;
    int input_cols_number;

    std::vector<std::vector<bool>> getCombinations(int n, int k) const;
    uvec polinomToIndexes(std::vector<bool> polinom) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    COMBI();
    int save(std::string path) const override;
    int load(const std::string& path) override;
    double predict(rowvec x) const override;
    vec predict(mat x) const override;
    COMBI& fit(mat x, vec y, const Criterion& criterion) override;
};


mat polynomailFeatures(const mat X, int max_degree);

}