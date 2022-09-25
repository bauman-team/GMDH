#include "combi.h"

namespace GMDH
{
	VectorVu16 COMBI::generateCombinations(int n_cols) const { 
		return nChooseK(n_cols, level);
	}

	GmdhModel& COMBI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, double testSize,
						  int pAverage, int threads, int verbose, double limit) {
		validateInputData(&testSize, &pAverage, &threads);
		return GmdhModel::gmdhFit(x, y, criterion, pAverage, testSize, pAverage, threads, verbose, limit);
	}
}