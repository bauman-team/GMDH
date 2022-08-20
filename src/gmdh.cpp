#include <iostream>
#include "gmdh.h"

namespace GMDH {

    void GMDH::polinomialsEvaluation(const SplittedData& data, const Criterion& criterion, 
        IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, bool verbose) const {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            auto pairCoeffsEvaluation = criterion.calculate(data.xTrain(Eigen::all, (*beginCoeffsVec).combination()),
                                                            data.xTest(Eigen::all, (*beginCoeffsVec).combination()),
                                                            data.yTrain, data.yTest);
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
            if (unlikely(verbose))
                --(*leftTasks);
        }      
    }

    bool GMDH::nextLevelCondition(double &lastLevelEvaluation, uint8_t p, VectorC& combinations, const Criterion& criterion, const SplittedData& data) {
        VectorC _bestCombinations = getBestCombinations(combinations, kBest);

        // TODO: add threads or kBest value will be always small?
        if (criterion.getClassName() == "SequentialCriterion") {
            for (auto combBegin = std::begin(_bestCombinations), combEnd = std::end(_bestCombinations); combBegin != combEnd; ++combBegin) {
                auto pairCoeffsEvaluation = static_cast<const SequentialCriterion&>(criterion).recalculate(
                                            data.xTrain(Eigen::all, (*combBegin).combination()),
                                            data.xTest(Eigen::all, (*combBegin).combination()),
                                            data.yTrain, data.yTest, (*combBegin).bestCoeffs());
                (*combBegin).setEvaluation(pairCoeffsEvaluation.first);
            }
            std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        }

        double currLevelEvaluation = getMeanCriterionValue(_bestCombinations, p);
        if (lastLevelEvaluation > currLevelEvaluation) {
            bestCombinations = std::move(_bestCombinations);
            lastLevelEvaluation = currLevelEvaluation;
            return true;
        }
        return false;
    }
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
    std::string GMDH::getModelName() const
    {
        std::string modelName = std::string(boost::typeindex::type_id_runtime(*this).pretty_name());
        modelName = modelName.substr(modelName.find_last_of(':') + 1);
        return modelName;
    }

    VectorC GMDH::getBestCombinations(VectorC& combinations, int k) const
    {
        k = std::min(k, static_cast<int>(combinations.size()));
        VectorC _bestCombinations(std::begin(combinations), std::begin(combinations) + k);
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

    double GMDH::getMeanCriterionValue(const VectorC& sortedCombinations, int k) const
    {
        k = std::min(k, static_cast<int>(sortedCombinations.size()));
        double currLevelEvaluation = 0;
        for (auto combBegin = std::begin(sortedCombinations); combBegin != std::begin(sortedCombinations) + k; ++combBegin)
            currLevelEvaluation += (*combBegin).evaluation();
        currLevelEvaluation /= static_cast<double>(k);
        return currLevelEvaluation;
    }

    GMDH& GMDH::fit(MatrixXd x, VectorXd y, const Criterion& criterion, double testSize, bool shuffle, int randomSeed, 
                    uint8_t p, int threads, int verbose) { // TODO: except threads, p = 0 error!!!

        using namespace indicators;
        using T = boost::packaged_task<void>;

        std::unique_ptr<ProgressBar> progressBar;

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
        auto lastLevelEvaluation = std::numeric_limits<double>::max();

        MatrixXd modifiedX(x.rows(), x.cols() + 1); 
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;

        SplittedData data = splitData(modifiedX, y, testSize, shuffle, randomSeed);

        /*std::cout << data.xTrain << "\n\n";
        std::cout << data.xTest << "\n\n";
        std::cout << data.yTrain << "\n\n";
        std::cout << data.yTest << "\n\n";*/

        while (level < data.xTrain.cols()) { // TODO: move condition to virtual method
            VectorC evaluationCoeffsVec;
            auto combinations = getCombinations(data.xTrain.cols() - 1, level);
            evaluationCoeffsVec.resize(combinations.size());
            auto currLevelEvaluation = std::begin(evaluationCoeffsVec);
            for (auto it = std::begin(combinations); it != std::end(combinations); ++it, ++currLevelEvaluation)
                currLevelEvaluation->setCombination(std::move(*it));

            leftTasks = static_cast<int>(evaluationCoeffsVec.size());

            if (verbose) {
                progressBar = std::make_unique<ProgressBar>(
                    option::BarWidth{30},
                    option::Start{"LEVEL " + std::to_string(level) + " (" + std::to_string(evaluationCoeffsVec.size())  + " combinations) ["},
                    option::End{"]"},
                    option::ShowElapsedTime{true},
                    option::ShowPercentage{true},
                    option::Lead{">"}
                );
                show_console_cursor(false);
                progressBar->set_progress(0);
            }
            decltype(auto) model = this;
            auto combsPortion = static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads)));
            for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) { 
                boost::packaged_task<void> pt([model=static_cast<const GMDH*>(model), &data=static_cast<const SplittedData&>(data), 
                &criterion=static_cast<const Criterion&>(criterion), &evaluationCoeffsVec, &leftTasks, verbose, combsPortion, i] () { 
                    model->polinomialsEvaluation(data, criterion, std::begin(evaluationCoeffsVec) + combsPortion * i,
                    std::begin(evaluationCoeffsVec) + std::min(static_cast<size_t>(combsPortion * (i + 1)), evaluationCoeffsVec.size()), 
                    &leftTasks, verbose);});
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
            } else {
                boost::when_all(futures.begin(), futures.end()).get();
            }

            // > or >= ?
            if (nextLevelCondition(lastLevelEvaluation, p, evaluationCoeffsVec, criterion, data)) { 
                ++level; //TODO: add goToTheNextLevel() virtual method
                futures.clear();
            }
            else {
                show_console_cursor(true);
                break; // TODO: change condition of ending cycle
            }

        }

        return *this;   
    }

    /*mat polynomailFeatures(const mat X, int max_degree) {
        int n = X.n_cols;
        std::vector <int> d(n);
        std::iota(d.begin(), d.end(), 0);
        mat poly_X;
        std::vector<std::vector<int>> monoms;
        for (int degree = 1; degree <= max_degree; ++degree)
        {
            std::vector <int> v(degree + 1, 0);
            while (1) {
                for (int i = 0; i < degree; i++) {
                    if (v[i] >= n) {
                        v[i + 1] += 1;
                        for (int k = i; k >= 0; k--) v[k] = v[i + 1];
                    }
                }
                if (v[degree] > 0) break;

                vec column(X.n_rows, fill::ones);
                std::vector<int> monom;
                for (int i = degree - 1; i > -1; i--)
                {
                    column %= X.col(d[v[i]]);
                    monom.push_back(d[v[i]]);
                }
                //column.print();
                poly_X.insert_cols(poly_X.n_cols, column);
                monoms.push_back(monom);
                v[0]++;
            }
        }
        return poly_X;
    }*/

    PairMVXd convertToTimeSeries(VectorXd x, int lags)
    {
        VectorXd yTimeSeries = x.tail(x.size() - lags);
        MatrixXd xTimeSeries(x.size() - lags, lags);
        for (int i = 0; i < x.size() - lags; ++i)
            xTimeSeries.row(i) = x.segment(i, lags);
        return PairMVXd(xTimeSeries, yTimeSeries);
    }

    SplittedData splitData(MatrixXd x, VectorXd y, double testSize, bool shuffle, int randomSeed)
    {
        SplittedData data;
        if (!shuffle)
        {
            data.xTrain = x.topRows(x.rows() - round(x.rows() * testSize));
            data.xTest = x.bottomRows(round(x.rows() * testSize));
            data.yTrain = y.head(y.size() - round(y.size() * testSize));
            data.yTest = y.tail(round(y.size() * testSize));
        }
        else
        {
            if (randomSeed != 0)
                std::srand(randomSeed);
            else
                std::srand(std::time(NULL));

            VectorI shuffled_rows_indexes(x.rows());
            std::iota(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end(), 0);
            std::random_shuffle(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end());

            VectorI train_indexes(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end() - round(x.rows() * testSize));
            VectorI test_indexes(shuffled_rows_indexes.end() - round(x.rows() * testSize), shuffled_rows_indexes.end());

            data.xTrain = x(train_indexes, Eigen::all);
            data.xTest = x(test_indexes, Eigen::all);
            data.yTrain = y(train_indexes);
            data.yTest = y(test_indexes);
        }
        return data;
    }

}

