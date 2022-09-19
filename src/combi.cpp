#include "combi.h"

namespace GMDH
{
	VectorVu16 COMBI::generateCombinations(int n_cols) const { // TODO: maybe change for bit masks 
		return nChooseK(n_cols, level);
	}

	GmdhModel& COMBI::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, double testSize,
						  uint8_t pAverage, int threads, int verbose, double limit) {
		validateInputData(&testSize, &pAverage, &threads);
		return GmdhModel::gmdhFit(x, y, criterion, pAverage, testSize, pAverage, threads, verbose, limit);
	}
}