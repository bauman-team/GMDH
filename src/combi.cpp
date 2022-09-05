#include "combi.h"

namespace GMDH
{
	VectorVu16 COMBI::generateCombinations(int n_cols) const { // TODO: maybe change for bit masks 
		return nChooseK(n_cols, level);
	}

	GmdhModel& COMBI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, double testSize,
				     bool shuffle, int randomSeed, uint8_t pAverage, int threads, int verbose, double limit) {
		return GmdhModel::fit(x, y, criterion, pAverage, testSize, shuffle, randomSeed, pAverage, threads, verbose, limit);
	}
}