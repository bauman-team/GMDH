#include "multi.h"

namespace GMDH
{
	std::vector<std::vector<uint16_t>> MULTI::getCombinations(int n_cols, int level) const // TODO: maybe change for bit masks
	{
		std::vector<std::vector<uint16_t>> combs;
		if (level == 1)
			return COMBI::getCombinations(n_cols, level);
	
		for (auto comb : bestCombinations)
		{
			for (int i = 0; i < n_cols; ++i)
			{
				std::vector<uint16_t> temp(comb.combination());
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
}
