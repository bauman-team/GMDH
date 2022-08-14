#include <iostream>
#include "gmdh.h"

namespace GMDH {

    void GMDH::polinomialsEvaluation(const MatrixXd& x, const VectorXd& y, // TODO: typedef (using) for all types
    const Criterion& criterion, std::vector<Combination>::iterator beginCoeffsVec, std::vector<Combination>::iterator endCoeffsVec) const {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            //std::vector<int> colsIndexes = polynomialToIndexes(*beginComb); 
            auto pairCoeffsEvaluation = criterion.calculate(x(Eigen::all, (*beginCoeffsVec).combination()), y);
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
        }      
    }

    bool GMDH::nextLevelCondition(double &lastLevelEvaluation, uint8_t p, std::vector<Combination>& combinations) {
        std::vector<Combination> _bestCombinations; // TODO: maybe change multimap to sort vector
        std::pair<int, Combination> bestComb;//(0, combinations[0]);
        for (auto i = 0; i < p; ++i) {
            bestComb = std::pair<int, Combination>(0, combinations[0]);
            for (auto j = 0; j != combinations.size(); ++j) 
                if (combinations[j].evaluation() < bestComb.second.evaluation()) {  // < or <= ?
                    bestComb.first = j;
                    bestComb.second = combinations[j];
                }
            combinations.erase(std::begin(combinations) + bestComb.first); // TODO: optimization
            _bestCombinations.push_back(bestComb.second);
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

    std::string GMDH::getModelName() const
    {
        std::string modelName = std::string(boost::typeindex::type_id_runtime(*this).pretty_name());
        modelName = modelName.substr(modelName.find_last_of(':') + 1);
        return modelName;
    }

    GMDH& GMDH::fit(MatrixXd x, VectorXd y, const Criterion& criterion, uint8_t p, int threads, int verbose) { // TODO: except threads = 0 error!!!
        using namespace indicators;
        ProgressBar progressBar {
              option::BarWidth{30},
              option::End{"]"},
              option::ShowElapsedTime{true},
              option::ShowPercentage{true},
              option::Lead{">"}
        };
        /*
        boost::function<void(const MatrixXd&, const VectorXd&, const Criterion&, std::vector<std::vector<bool> >::const_iterator,
        std::vector<std::vector<bool> >::const_iterator, std::vector<std::pair<std::pair<double, VectorXd>, 
        std::vector<bool> >>::iterator)> calcEvaluationCoeffs = polinomialsEvaluation;
        */
        level = 1;
        if (threads == -1)
            threads = boost::thread::hardware_concurrency();
        else
            threads = std::min(threads, static_cast<int>(boost::thread::hardware_concurrency())); // TODO: change limit
        boost::asio::thread_pool pool(threads);

        inputColsNumber = x.cols();
        auto lastLevelEvaluation = std::numeric_limits<double>::max();

        MatrixXd modifiedX(x.rows(), x.cols() + 1); 
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;

        while (level < modifiedX.cols()) { // TODO: move condition to method
            // TODO: add using (as typedef)
            std::vector<Combination> evaluationCoeffsVec; 
            auto combinations = getCombinations(x.cols(), level);
            evaluationCoeffsVec.resize(combinations.size());
            auto currLevelEvaluation = std::begin(evaluationCoeffsVec);
            for (auto it = std::begin(combinations); it != std::end(combinations); ++it)
                currLevelEvaluation->setCombination(std::move(*it));

            if (verbose) {
                progressBar.set_option(option::Start{"LEVEL " + std::to_string(level) + " (" + std::to_string(evaluationCoeffsVec.size())  + " combinations) ["});
                show_console_cursor(false);
                progressBar.set_progress(0);
            }

            using T = boost::packaged_task<void>;
            std::vector<boost::unique_future<T::result_type> > futures; // TODO: reserve??? or array

            auto combsPortion = static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads)));
            for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) {
                boost::packaged_task<void> pt(boost::bind(&GMDH::polinomialsEvaluation, this, modifiedX, y, boost::ref(criterion), // MAYBE boost::function<>
                    evaluationCoeffsVec.begin() + combsPortion * i,
                    evaluationCoeffsVec.begin() + std::min(static_cast<size_t>(combsPortion * (i + 1)), evaluationCoeffsVec.size())));
                futures.push_back(pt.get_future());
                post(pool, std::move(pt));
            }
            boost::when_all(futures.begin(), futures.end()).get();

            // > or >= ?
            if (nextLevelCondition(lastLevelEvaluation, p, evaluationCoeffsVec)) { 
                ++level; //TODO: add goToTheNextLevel() virtual method
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

    std::pair<MatrixXd, VectorXd> convertToTimeSeries(VectorXd x, int lags)
    {
        VectorXd yTimeSeries = x.tail(x.size() - lags);
        MatrixXd xTimeSeries(x.size() - lags, lags);
        for (int i = 0; i < x.size() - lags; ++i)
            xTimeSeries.row(i) = x.segment(i, lags);
        return std::pair<MatrixXd, VectorXd>(xTimeSeries, yTimeSeries);
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

        
        std::vector<int> shuffled_rows_indexes(x.rows());
        std::iota(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end(), 0);
        std::random_shuffle(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end());

        std::vector<int> train_indexes(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end() - round(x.rows() * testSize));
        std::vector<int> test_indexes(shuffled_rows_indexes.end() - round(x.rows() * testSize), shuffled_rows_indexes.end());
        
        SplittedData data;
        data.xTrain = x(train_indexes, Eigen::all);
        data.xTest = x(test_indexes, Eigen::all);
        data.yTrain = y(train_indexes);
        data.yTest = y(test_indexes);

        return data;
    }

}

