#include <iostream>
#include "gmdh.h"

namespace GMDH {

    void GMDH::polinomialsEvaluation(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, 
        IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, bool verbose) const {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            auto pairCoeffsEvaluation = criterion.calculate(x(Eigen::all, (*beginCoeffsVec).combination()), y);
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
            if (unlikely(verbose))
                --(*leftTasks);
        }      
    }

    bool GMDH::nextLevelCondition(double &lastLevelEvaluation, uint8_t p, VectorC& combinations) {
        VectorC _bestCombinations(std::begin(combinations), std::begin(combinations) + p); 
        std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
        for (auto combBegin = std::begin(combinations) + p, combEnd = std::end(combinations); 
        combBegin != combEnd; ++combBegin) {
            if (*combBegin < _bestCombinations.back()) {
                std::swap(*combBegin, _bestCombinations.back());
                std::sort(std::begin(_bestCombinations), std::end(_bestCombinations));
            }
        }
        double currLevelEvaluation = 0;
        for (auto i : _bestCombinations)
            currLevelEvaluation += i.evaluation();
        currLevelEvaluation /= static_cast<double>(p);
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

    GMDH& GMDH::fit(MatrixXd x, VectorXd y, const Criterion& criterion, uint8_t p, int threads, int verbose) { // TODO: except threads, p = 0 error!!!
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

        while (level < modifiedX.cols()) { // TODO: move condition to virtual method
            VectorC evaluationCoeffsVec;
            auto combinations = getCombinations(x.cols(), level);
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

            auto combsPortion = static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads)));
            for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) {
                boost::packaged_task<void> pt(boost::bind(&GMDH::polinomialsEvaluation, this, modifiedX, y, boost::ref(criterion), 
                    std::begin(evaluationCoeffsVec) + combsPortion * i,
                    std::begin(evaluationCoeffsVec) + std::min(static_cast<size_t>(combsPortion * (i + 1)), evaluationCoeffsVec.size()), &leftTasks, verbose));
                futures.push_back(pt.get_future());
                post(pool, std::move(pt));
                //if (verbose)
                  //  progressBar->set_progress(100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size());
            }

            if (verbose) {
                while (leftTasks) {
                    progressBar->set_progress(100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size());
                }
                progressBar->set_progress(100);
            } else {
                boost::when_all(futures.begin(), futures.end()).get();
            }

            // > or >= ?
            if (nextLevelCondition(lastLevelEvaluation, p, evaluationCoeffsVec)) { 
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

    SplittedData splitTimeSeries(MatrixXd x, VectorXd y, double testSize)
    {
        SplittedData data;
        data.xTrain = x.topRows(x.rows() - round(x.rows() * testSize));
        data.xTest = x.bottomRows(round(x.rows() * testSize));
        data.yTrain = y.head(y.size() - round(y.size() * testSize));
        data.yTest = y.tail(round(y.size() * testSize));
        return data;
    }

    SplittedData splitData(MatrixXd x, VectorXd y, double testSize, bool shuffle, int randomSeed)
    {
        if (!shuffle)
            return splitTimeSeries(x, y, testSize);

        if (randomSeed != 0)
            std::srand(randomSeed);
        else
            std::srand(std::time(NULL));

        
        VectorI shuffled_rows_indexes(x.rows());
        std::iota(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end(), 0);
        std::random_shuffle(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end());

        VectorI train_indexes(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end() - round(x.rows() * testSize));
        VectorI test_indexes(shuffled_rows_indexes.end() - round(x.rows() * testSize), shuffled_rows_indexes.end());
        
        SplittedData data;
        data.xTrain = x(train_indexes, Eigen::all);
        data.xTest = x(test_indexes, Eigen::all);
        data.yTrain = y(train_indexes);
        data.yTest = y(test_indexes);

        return data;
    }

}

