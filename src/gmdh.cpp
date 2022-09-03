#include "gmdh.h"
#include <stdio.h>
namespace GMDH {

/*
    int GMDH::calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator >> beginTasksVec, 
    const std::vector<std::shared_ptr<std::vector<Combination>::iterator >> endTasksVec) const {
        auto leftTasks = 0;
        auto itBegComb = std::cbegin(beginTasksVec), itEndComb = std::cbegin(endTasksVec);
        for (;itBegComb != std::cend(beginTasksVec); ++itBegComb, ++itEndComb)
            leftTasks += (**itEndComb) - (**itBegComb);
        return leftTasks;
    }
*/
    std::string GMDH::getModelName() const {
        auto modelName{ boost::typeindex::type_id_runtime(*this).pretty_name() };
        modelName = modelName.substr(modelName.find_last_of(':') + 1);
        return modelName;
    }

    VectorVu16 GMDH::nChooseK(int n, int k) const {
        struct c_unique {
            uint16_t current;
            c_unique() { current = -1; }
            uint16_t operator()() { return ++current; }
        } UniqueNumber;

        VectorVu16 combs;
        VectorU16 comb(k);
        auto first{ std::begin(comb) }, last{ std::end(comb) };

        std::generate(first, last, UniqueNumber);
        combs.push_back(comb);

        while ((*first) != n - k) {
            auto mt{ last };
            while (*(--mt) == n - (last - mt));
            (*mt)++;
            while (++mt != last) *mt = *(mt - 1) + 1;
            combs.push_back(comb);
        }

        for (auto i = 0; i < combs.size(); ++i)
            combs[i].push_back(n);
        return combs;
    }

    VectorC GMDH::getBestCombinations(VectorC& combinations, int k) const {
        k = std::min(k, static_cast<int>(combinations.size()));
        VectorC _bestCombinations{ std::begin(combinations), std::begin(combinations) + k };
        std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        for (auto combBegin = std::begin(combinations) + k, combEnd = std::end(combinations);
            combBegin != combEnd; ++combBegin) {
            if (*combBegin < _bestCombinations.back()) {
                std::swap(*combBegin, _bestCombinations.back());
                std::sort(std::begin(_bestCombinations), std::end(_bestCombinations)); 
            }
        }
        return _bestCombinations;
    }

    double GMDH::getMeanCriterionValue(const VectorC& sortedCombinations, int k) const {
        k = std::min(k, static_cast<int>(sortedCombinations.size()));
        auto currLevelEvaluation{ 0. };
        for (auto combBegin = std::cbegin(sortedCombinations); combBegin != std::cbegin(sortedCombinations) + k; ++combBegin)
            currLevelEvaluation += (*combBegin).evaluation();
        currLevelEvaluation /= static_cast<double>(k);
        return currLevelEvaluation;
    }

    void GMDH::polynomialsEvaluation(const SplittedData& data, const Criterion& criterion, 
        IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, bool verbose) const {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            auto pairCoeffsEvaluation{ criterion.calculate(data.xTrain(Eigen::all, (*beginCoeffsVec).combination()),
                                                            data.xTest(Eigen::all, (*beginCoeffsVec).combination()),
                                                            data.yTrain, data.yTest) };
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
            if (unlikely(verbose))
                --(*leftTasks); 
        }      
    }

    bool GMDH::nextLevelCondition(double &lastLevelEvaluation, int kBest, uint8_t pAverage, VectorC& combinations, 
                                  const Criterion& criterion, SplittedData& data, double limit) {
        auto _bestCombinations{ getBestCombinations(combinations, kBest) };
        if (criterion.getClassName() == "SequentialCriterion") { 
            // TODO: add threads or kBest value will be always small?
            for (auto comb : _bestCombinations) {
                auto pairCoeffsEvaluation{ static_cast<const SequentialCriterion&>(criterion).recalculate( // TODO: THE WORST code style 
                                            data.xTrain(Eigen::all, comb.combination()),
                                            data.xTest(Eigen::all, comb.combination()),
                                            data.yTrain, data.yTest, comb.bestCoeffs()) };
                comb.setEvaluation(pairCoeffsEvaluation.first);
            }
            std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        }
        auto currLevelEvaluation{ getMeanCriterionValue(_bestCombinations, pAverage) };
        //std::cout << "\n" << currLevelEvaluation << "\n";

        if (lastLevelEvaluation - currLevelEvaluation > limit) {
            bestCombinations[0] = std::move(_bestCombinations);
            lastLevelEvaluation = currLevelEvaluation;
            if (++level < data.xTrain.cols())
                return true;
        }
        return false;
    }

    GMDH& GMDH::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, double testSize, 
                    bool shuffle, int randomSeed, uint8_t pAverage, int threads, int verbose, double limit) {

        using namespace indicators;
        using T = boost::packaged_task<void>;
        std::unique_ptr<ProgressBar> progressBar;
        validateInputData(&testSize, &pAverage, &threads);
        
        level = 1;
        if (threads == -1)
            threads = boost::thread::hardware_concurrency(); // TODO: maybe find optimal count based on data.size() and hardware_concurrency()
        else 
            threads = std::min(threads, static_cast<int>(boost::thread::hardware_concurrency())); // TODO: change limit

        boost::asio::thread_pool pool(threads); 
        std::vector<boost::unique_future<T::result_type> > futures;
        futures.reserve(threads);
        std::atomic<int> leftTasks; // TODO: change to volatile structure

        inputColsNumber = x.cols();
        auto lastLevelEvaluation{ std::numeric_limits<double>::max() };

        MatrixXd modifiedX{ x.rows(), x.cols() + 1 };
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        auto data{ splitData(modifiedX, y, testSize, shuffle, randomSeed) };
        modifiedX.resize(0, 0); // TODO: clear???

        /*std::cout << data.xTrain << "\n\n";
        std::cout << data.xTest << "\n\n";
        std::cout << data.yTrain << "\n\n";
        std::cout << data.yTest << "\n\n";*/

        VectorC evaluationCoeffsVec;
        do {
            futures.clear();
            evaluationCoeffsVec.clear();
            auto combinations{ generateCombinations(data.xTrain.cols() - 1) };
            evaluationCoeffsVec.resize(combinations.size());
            auto currLevelEvaluation{ std::begin(evaluationCoeffsVec) };
            for (auto it = std::begin(combinations); it != std::end(combinations); ++it, ++currLevelEvaluation)
                currLevelEvaluation->setCombination(std::move(*it));

            leftTasks = static_cast<int>(evaluationCoeffsVec.size());
            if (verbose) {
                progressBar = std::make_unique<ProgressBar>(
                    option::BarWidth{ 30 },
                    option::Start{ "LEVEL " + std::to_string(level) + ((std::to_string(level).size() == 1) ? "  [" : " [")},
                    option::End{ "]" },
                    option::Lead{ ">" },
                    option::ShowElapsedTime{ true },
                    option::ShowPercentage{ true },
                    option::PostfixText{ "(" + std::to_string(evaluationCoeffsVec.size()) + " combinations)" }
                );
                show_console_cursor(false);
                progressBar->set_progress(0);
            }
            decltype(auto) model = this;
            auto combsPortion{ static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads))) };
            for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) {
                boost::packaged_task<void> pt([model = static_cast<const GMDH*>(model), 
                    &data = static_cast<const SplittedData&>(data), &criterion = static_cast<const Criterion&>(criterion), 
                    &evaluationCoeffsVec, &leftTasks, verbose, combsPortion, i]() {
                        model->polynomialsEvaluation(data, criterion, std::begin(evaluationCoeffsVec) + combsPortion * i,
                            std::begin(evaluationCoeffsVec) + std::min(static_cast<size_t>(combsPortion * (i + 1)), 
                            evaluationCoeffsVec.size()), &leftTasks, verbose); });
                futures.push_back(pt.get_future());
                post(pool, std::move(pt));
            }

            if (verbose) {
                while (leftTasks) {
                    if (progressBar->current() < 100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size())
                        progressBar->set_progress(100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size());
                    boost::this_thread::sleep_for(boost::chrono::milliseconds(20));
                }
                progressBar->set_progress(100);
            }
            else
                boost::when_all(std::begin(futures), std::end(futures)).get();
        } while (nextLevelCondition(lastLevelEvaluation, kBest, pAverage, evaluationCoeffsVec, criterion, data, limit));
        if (verbose)
            show_console_cursor(true);
        return *this;   
    }

    int validateInputData(double *testSize, uint8_t *pAverage, int *threads) {
        auto errorCode{ 0 };
#ifdef GMDH_LIB
        std::cout << DISPLAYEDCOLORWARNING;
#elif GMDH_MODULE
        pybind11::scoped_ostream_redirect stream(
            std::cout,                               // std::ostream&
            pybind11::module_::import("sys").attr("stdout") // Python output
        );
#endif
        if (*testSize <= 0) { // TODO: add range 
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("value of testSize","testSize = 0.5");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("value of testSize","testSize = 0.5"), 1);
#endif
            *testSize = 0.5;
            errorCode |= 1;
        }
        if (threads && (*threads < -1 || !*threads))
        {
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("number of threads","threads = 1");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("number of threads","threads = 1"), 1);
#endif
            *threads = 1;
            errorCode |= 2;
        }
        if (pAverage && !(*pAverage)) {
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("number of pAverage","pAverage = 1");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("number of pAverage","pAverage = 1"), 1);
#endif
            *pAverage = 1;
            errorCode |= 4;
        }
#ifdef GMDH_LIB
            std::cout << DISPLAYEDCOLORINFO;
#endif
        return errorCode;
    }

    PairMVXd timeSeriesTransformation(VectorXd x, int lags) {
        VectorXd yTimeSeries{ x.tail(x.size() - lags) };
        MatrixXd xTimeSeries{ x.size() - lags, lags };
        for (auto i = 0; i < x.size() - lags; ++i)
            xTimeSeries.row(i) = x.segment(i, lags);
        return { xTimeSeries, yTimeSeries };
    }

    SplittedData splitData(const MatrixXd& x, const VectorXd& y, double testSize, bool shuffle, int randomSeed) {
        validateInputData(&testSize);   
        SplittedData data;
        if (!shuffle) {
            data.xTrain = x.topRows(x.rows() - round(x.rows() * testSize));
            data.xTest = x.bottomRows(round(x.rows() * testSize));
            data.yTrain = y.head(y.size() - round(y.size() * testSize));
            data.yTest = y.tail(round(y.size() * testSize));
        } else {
            if (randomSeed != 0) std::srand(randomSeed);
            else std::srand(std::time(NULL));

            VectorI shuffled_rows_indexes(x.rows());
            std::iota(std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes), 0);
            std::random_shuffle(std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes));

            VectorI train_indexes{ std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes) - round(x.rows() * testSize) };
            VectorI test_indexes{ std::end(shuffled_rows_indexes) - round(x.rows() * testSize), std::end(shuffled_rows_indexes) };

            data.xTrain = x(train_indexes, Eigen::all);
            data.xTest = x(test_indexes, Eigen::all);
            data.yTrain = y(train_indexes);
            data.yTest = y(test_indexes);
        }
        return data;
    }

    std::string Combination::getInfoForSaving() const {
        std::stringstream info;
        info.precision(12); // TODOL maybe change precision
        for (auto i : _combination) info << i << ' ';
        info << "\n";
        for (auto i : _bestCoeffs) info << i << ' ';
        info << "\n";
        return info.str();
    }
}

