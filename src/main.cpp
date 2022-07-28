#include <iostream>
#include <algorithm>
#include <vector>
#include <armadillo>

unsigned long nChoosek(unsigned long n, unsigned long k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    unsigned long result = n;
    for(unsigned long i = 2; i <= k; ++i) {
        result /= i;
        result *= (n - i + 1);
    }
    return result;
}

int main() {
    using namespace std;
    using namespace arma;

    cout << "Armadillo version: " << arma_version::as_string() << endl;
    //std::cout << nChoosek(100, 40) << "\n";
    int n, k;
    std::cin >> n >> k;

    std::vector<bool> v(n);
    std::fill(v.begin(), v.begin() + k, true);
    do {
        for (int i = 0; i < n; ++i)
            std::cout << v[i];
        std::cout << "\n";
    } while (std::prev_permutation(v.begin(), v.end()));
    cin >> n;
    return 0;
}
namespace GMDH {

enum class ExternalCriterion {regularity};  // TODO: define and realise external criterions

class GMDH {
    double ExternalCriterion() const;

public:
    virtual void save() const = 0;
    virtual int load() = 0;
    virtual GMDH& fit() = 0;
    virtual double predict() const = 0;
    
};

class COMBI : public GMDH {

    std::vector<bool> bestPolinom;
    std::vector<double> bestCoeffs;

    std::vector<double> InternalCriterion() const;
    
    
    
        
public:
    

    void save() const override;
    int load() override;
    COMBI& fit() override;
    double predict() const override;
};

}