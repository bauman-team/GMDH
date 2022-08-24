//#include "pch.h"
#include "multi.h"

namespace GMDH
{
	VectorVu16 MULTI::getCombinations(int n_cols, int level) const // TODO: maybe change for bit masks
	{
		VectorVu16 combs;
		if (level == 1)
			return COMBI::getCombinations(n_cols, level);
	
		for (auto comb : bestCombinations)
		{
			for (int i = 0; i < n_cols; ++i)
			{
				VectorU16 temp(comb.combination());
				if (std::find(temp.begin(), temp.end(), i) == temp.end())
				{
					temp.push_back(i);
					std::sort(temp.begin(), temp.end());
					if (std::find(combs.begin(), combs.end(), temp) == combs.end())
						combs.push_back(temp);
				}
			}
		}
		return combs;
	}

	GMDH& MULTI::fit(MatrixXd x, VectorXd y, Criterion& criterion, int _kBest, double testSize, bool shuffle, int randomSeed, uint8_t p, int threads, int verbose)
	{
		kBest = _kBest;
		return GMDH::fit(x, y, criterion, testSize, shuffle, randomSeed, p, threads, verbose);
	}
}
