#include <vector>
#include <armadillo>
#include <cmath>
#include <numeric>
#include <cstdint>


namespace GMDH {

using namespace arma;

class Criterion {

protected:
    vec internalCriterion(mat x_train, vec y_train) const;

public:
    virtual double calculate(mat x, vec y_real) = 0;
};

class RegularityCriterion : public Criterion {

    double test_size;

public:
    RegularityCriterion(double test_size = 0.33);
    double calculate(mat x, vec y_real) override;
};


class GMDH {

protected:
    int level;

public:
    GMDH();
    virtual void save() const = 0;
    virtual int load() = 0;
    virtual GMDH& fit(mat x, vec y, Criterion* criterion) = 0;
    virtual double predict() const = 0;
};

class COMBI : public GMDH {

    std::vector<bool> bestPolinom;
    vec bestCoeffs;

    std::vector<std::vector<bool>> get_combinations(int n, int k) const;
    //unsigned long nChoosek(unsigned long n, unsigned long k);
        
public:
    void save() const override;
    int load() override;
    double predict() const override;
    COMBI& fit(mat x, vec y, Criterion* criterion) override;
};


mat polynomail_features(const mat X, int max_degree);

}