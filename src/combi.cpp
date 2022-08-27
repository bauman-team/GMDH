//#include "pch.h"
#include "combi.h"

namespace GMDH
{
	VectorVu16 COMBI::getCombinations(int n_cols) const // TODO: maybe change for bit masks
	{
		return nChooseK(n_cols, level);
	}

	GMDH& COMBI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, double testSize, bool shuffle, int randomSeed, uint8_t p, int threads, int verbose)
	{
		return GMDH::fit(x, y, criterion, p, testSize, shuffle, randomSeed, p, threads, verbose);
	}
}