#include "gmdh.h"

namespace GMDH {
unsigned long COMBI::nChoosek(unsigned long n, unsigned long k)
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

}