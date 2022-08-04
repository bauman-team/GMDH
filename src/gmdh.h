#include <vector>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <utility>
//#include <unordered_map>


namespace GMDH {

using namespace arma;

class Criterion {

protected:
    vec internalCriterion(mat x_train, vec y_train) const;

public:
    virtual std::pair<double, vec> calculate(mat x, vec y_real) const = 0;
};

class RegularityCriterion : public Criterion {

    double test_size;

public:
    RegularityCriterion(double _test_size = 0.33);
    std::pair<double, vec> calculate(mat x, vec y_real) const override;
};


class GMDH {

protected:
    int level;

public:
    GMDH();
    virtual void save() const = 0;
    virtual int load() = 0;
    virtual GMDH& fit(mat x, vec y, const Criterion& criterion) = 0;
    virtual double predict() const = 0;
};

class COMBI : public GMDH { // TODO: split into separate files

    std::vector<bool> best_polinom;
    vec best_coeffs;

    std::vector<std::vector<bool>> getCombinations(int n, int k) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    void save() const override;
    int load() override;
    double predict() const override;
    COMBI& fit(mat x, vec y, const Criterion& criterion) override;
};


mat polynomailFeatures(const mat X, int max_degree);

}