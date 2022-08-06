#include <vector>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
//#include <unordered_map>


namespace GMDH {

using namespace arma;

struct splitted_data {
    mat x_train;
    mat x_test;
    vec y_train;
    vec y_test;
};

class Criterion {

protected:
    vec internalCriterion(mat x_train, vec y_train) const;

public:
    virtual std::pair<double, vec> calculate(mat x, vec y) const = 0;
};

class RegularityCriterionTS : public Criterion
{
protected:
    double test_size;

    std::pair<double, vec> getCriterionValue(splitted_data data) const;
public:
    RegularityCriterionTS(double _test_size = 0.33);
    std::pair<double, vec> calculate(mat x, vec y) const override;
};

class RegularityCriterion : public RegularityCriterionTS {

    bool shuffle;
    int random_seed;

public:
    RegularityCriterion(double _test_size = 0.33, bool _shuffle = true, int _random_seed = 0);
    std::pair<double, vec> calculate(mat x, vec y) const override;
};


class GMDH {

protected:
    int level;
    std::string model_name;

public:
    GMDH();
    virtual int save(std::string path) const = 0;
    virtual int load(std::string path) = 0;
    virtual GMDH& fit(mat x, vec y, const Criterion& criterion) = 0;
    virtual double predict(rowvec x) const = 0;
    virtual vec predict(mat x) const = 0;
    virtual std::string getBestPolymon() const = 0;
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
    int load(std::string path) override;
    double predict(rowvec x) const override;
    vec predict(mat x) const override;
    COMBI& fit(mat x, vec y, const Criterion& criterion) override;
    std::string getBestPolymon() const override;
};


mat polynomailFeatures(const mat X, int max_degree);
std::pair<mat, vec> convertToTimeSeries(vec x, int lags);
splitted_data splitTsData(mat x, vec y, double validate_size = 0.2);
splitted_data splitData(mat x, vec y, double validate_size = 0.2, bool shuffle = true, int _random_seed = 0);

}