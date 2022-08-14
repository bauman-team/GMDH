#include "combi.h"

namespace GMDH {

int COMBI::save(const std::string& path) const
{
    std::ofstream modelFile;
    modelFile.open(path);
    if (!modelFile.is_open())
        return -1;
    else {
        modelFile << getModelName() << "\n";
        modelFile << inputColsNumber << "\n";
        for (auto i : bestCombinations[0].combination()) modelFile << i << ' ';
        modelFile << "\n";
        modelFile.precision(10); // TODOL maybe change precision
        for (auto i : bestCombinations[0].bestCoeffs()) modelFile << i << ' ';
        modelFile << "\n";
        modelFile.close();
    }
    return 0;
}

int COMBI::load(const std::string& path)
{
    inputColsNumber = 0;
    bestCombinations.clear();
    bestCombinations.resize(1);
    std::vector<uint16_t> bestColsIndexes;

    std::ifstream modelFile;
    modelFile.open(path);
    if (!modelFile.is_open())
        return -1;
    else {
        std::string modelName;
        modelFile >> modelName;
        if (modelName != getModelName())
            return -1;
        else {
            (modelFile >> inputColsNumber).get();

            std::string colsIndexesLine;
            std::getline(modelFile, colsIndexesLine);
            std::stringstream indexStream(colsIndexesLine);
            uint16_t index;
            while (indexStream >> index)
                bestColsIndexes.push_back(index);
            bestCombinations[0].setCombination(std::move(bestColsIndexes));

            std::string coeffsLine;
            std::vector<double> coeffs;
            std::getline(modelFile, coeffsLine);
            std::stringstream coeffsStream(coeffsLine);
            double coeff;
            while (coeffsStream >> coeff)
                coeffs.push_back(coeff);
            bestCombinations[0].setBestCoeffs(Map<VectorXd>(coeffs.data(), coeffs.size()));
        }
        modelFile.close();
    }
    return 0;
}

double COMBI::predict(const RowVectorXd& x) const
{
    return predict(MatrixXd(x))[0];
}

VectorXd COMBI::predict(const MatrixXd& x) const
{
    MatrixXd modifiedX(x.rows(), x.cols() + 1);
    modifiedX.col(x.cols()).setOnes();
    modifiedX.leftCols(x.cols()) = x;
    return modifiedX(Eigen::all, bestCombinations[0].combination()) * bestCombinations[0].bestCoeffs();
}

std::vector<std::vector<uint16_t>> COMBI::getCombinations(int n, int k) const
{
    struct c_unique {
        uint16_t current;
        c_unique() { current = -1; }
        uint16_t operator()() { return ++current; }
    } UniqueNumber;

    std::vector<std::vector<uint16_t>> combs;
    std::vector<uint16_t> comb(k);
    std::vector<uint16_t>::iterator first = comb.begin(), last = comb.end();

    std::generate(first, last, UniqueNumber);
    combs.push_back(comb);

    while ((*first) != n - k) {
        std::vector<uint16_t>::iterator mt = last;
        while (*(--mt) == n - (last - mt));
        (*mt)++;
        while (++mt != last) *mt = *(mt - 1) + 1;
        combs.push_back(comb);
    }

    for (int i = 0; i < combs.size(); ++i)
        combs[i].push_back(n);
    return combs;
}

/*
COMBI& COMBI::fit(MatrixXd x, VectorXd y, const Criterion& criterion, int threads, int verbose)
{   
    level = 1;
    if (threads == -1)
        threads = boost::thread::hardware_concurrency();
    else
        threads = std::min(threads, static_cast<int>(boost::thread::hardware_concurrency()));
    boost::asio::thread_pool pool(threads);
    boost::function<void(const MatrixXd&, const VectorXd&, const Criterion&, std::vector<std::vector<bool> >::const_iterator,
    std::vector<std::vector<bool> >::const_iterator, std::vector<std::pair<std::pair<double, VectorXd>, 
    std::vector<bool> >>::iterator)> calcEvaluationCoeffs = 
        [polynomialToIndexes=&polynomialToIndexes] (const MatrixXd& x, const VectorXd& y, const Criterion& criterion, std::vector<std::vector<bool> >::const_iterator beginComb, 
        std::vector<std::vector<bool> >::const_iterator endComb, std::vector<std::pair<std::pair<double, VectorXd>, 
        std::vector<bool> >>::iterator beginCoeffsVec) {
            for (; beginComb < endComb; ++beginComb, ++beginCoeffsVec) {
                //std::vector<int> colsIndexes = polynomialToIndexes(*beginComb); // TODO: typedef (using) for all types
                //beginCoeffsVec->first = criterion.calculate(x(Eigen::all, colsIndexes), y);
                beginCoeffsVec->second = *beginComb;
            }      
        };

    double lastLevelEvaluation = std::numeric_limits<double>::max();
    std::vector<bool> bestPolynomial;
    inputColsNumber = x.cols();

    MatrixXd modifiedX(x.rows(), x.cols() + 1);
    modifiedX.col(x.cols()).setOnes();
    modifiedX.leftCols(x.cols()) = x;

    while (level < modifiedX.cols()) {
        std::vector<std::pair<std::pair<double, VectorXd>, std::vector<bool> >> evaluationCoeffsVec; 
        std::vector<std::pair<std::pair<double, VectorXd>, std::vector<bool> >>::const_iterator currLevelEvaluation; // TODO: add using (as typedef)
        std::vector<std::vector<bool>> combinations = getCombinations(x.cols(), level);

        using namespace indicators;
        ProgressBar progressBar {
              option::BarWidth{30},
              option::Start{"LEVEL " + std::to_string(level) + " (" + std::to_string(combinations.size())  + " combinations) ["},
              option::End{"]"},
              option::ShowElapsedTime{true},
              option::ShowPercentage{true},
              option::Lead{">"}
        };
        if (verbose) {
            show_console_cursor(false);
            progressBar.set_progress(0);
        }

        if (threads > 1) {
            using T = boost::packaged_task<void>;
            std::vector<boost::unique_future<T::result_type> > futures; // TODO: reserve??? or array

            evaluationCoeffsVec.resize(combinations.size());
            auto combsPortion = static_cast<int>(std::ceil(combinations.size() / static_cast<double>(threads)));
            for (auto i = 0; i * combsPortion < combinations.size(); ++i) {
                boost::packaged_task<void> pt(boost::bind(calcEvaluationCoeffs, modifiedX, y, boost::ref(criterion),
                    combinations.cbegin() + combsPortion * i,
                    combinations.cbegin() + std::min(static_cast<size_t>(combsPortion * (i + 1)), combinations.size()),
                    evaluationCoeffsVec.begin() + combsPortion * i));
                futures.push_back(pt.get_future());
                post(pool, std::move(pt));
            }
            boost::when_all(futures.begin(), futures.end()).get();
        }
        else {
            evaluationCoeffsVec.reserve(combinations.size());
            for (int i = 0; i < combinations.size(); ++i) {
                std::vector<int> colsIndexes = polynomialToIndexes(combinations[i]);
                evaluationCoeffsVec.push_back(std::pair<std::pair<double, VectorXd>, std::vector<bool> >(criterion.calculate(modifiedX(Eigen::all, colsIndexes), y), combinations[i]));
                if (verbose) {
                    progressBar.set_progress(100.0 * (i + 1) / combinations.size());
                }
            }
        }

        // > or >= ?
        if (lastLevelEvaluation > 
        (currLevelEvaluation = std::min_element(
        std::cbegin(evaluationCoeffsVec), 
        std::cend(evaluationCoeffsVec), 
        [](std::pair<std::pair<double, VectorXd>, std::vector<bool> > first, 
        std::pair<std::pair<double, VectorXd>, std::vector<bool> > second) { 
            return first.first.first < second.first.first;
        }))->first.first) {
            lastLevelEvaluation = currLevelEvaluation->first.first;
            bestPolynomial = currLevelEvaluation->second;
            bestCoeffs = currLevelEvaluation->first.second;
        }
        else {
            show_console_cursor(true);
            break; // TODO: change condition of ending cycle
        }
        ++level;
    }

    bestColsIndexes = polynomialToIndexes(bestPolynomial);

    return *this;
}
*/

std::string COMBI::getBestPolynomial() const
{
    std::string polynomialStr = "y =";
    auto bestColsIndexes = bestCombinations[0].combination();
    auto bestCoeffs = bestCombinations[0].bestCoeffs();
    for (int i = 0; i < bestColsIndexes.size(); ++i) {
        if (bestCoeffs[i] > 0) {
            if (i > 0)
                polynomialStr += " + ";
            else
                polynomialStr += " ";
        }
        else
            polynomialStr += " - ";
        polynomialStr += std::to_string(abs(bestCoeffs[i]));
        if (i != bestColsIndexes.size() - 1)
            polynomialStr += "*x" + std::to_string(bestColsIndexes[i] + 1);
    }
    return polynomialStr;
}
}