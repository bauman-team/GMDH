#include "gmdh.h"


namespace GMDH {

/*
    int GmdhModel::calculateLeftTasksForVerbose(const std::vector<std::shared_ptr<std::vector<Combination>::iterator >> beginTasksVec, 
    const std::vector<std::shared_ptr<std::vector<Combination>::iterator >> endTasksVec) const {
        auto leftTasks = 0;
        auto itBegComb = std::cbegin(beginTasksVec), itEndComb = std::cbegin(endTasksVec);
        for (;itBegComb != std::cend(beginTasksVec); ++itBegComb, ++itEndComb)
            leftTasks += (**itEndComb) - (**itBegComb);
        return leftTasks;
    }
*/
    std::string GmdhModel::getModelName() const {
        auto modelName{ boost::typeindex::type_id_runtime(*this).pretty_name() };
        modelName = modelName.substr(modelName.find_last_of(':') + 1);
        return modelName;
    }

    VectorVu16 GmdhModel::nChooseK(int n, int k) const {
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

    double GmdhModel::getMeanCriterionValue(const VectorC& sortedCombinations, int k) const {
        k = std::min(k, static_cast<int>(sortedCombinations.size()));
        auto currLevelEvaluation{ 0. };
        for (auto combBegin = std::cbegin(sortedCombinations); combBegin != std::cbegin(sortedCombinations) + k; ++combBegin)
            currLevelEvaluation += (*combBegin).evaluation();
        currLevelEvaluation /= static_cast<double>(k);
        return currLevelEvaluation;
    }

    void GmdhModel::polynomialsEvaluation(const SplittedData& data, const Criterion& criterion,
        IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, bool verbose) const {
        for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
            auto pairCoeffsEvaluation{ criterion.calculate(xDataForCombination(data.xTrain, (*beginCoeffsVec).combination()),
                                                           xDataForCombination(data.xTest, (*beginCoeffsVec).combination()),
                                                           data.yTrain, data.yTest) };
            (*beginCoeffsVec).setEvaluation(pairCoeffsEvaluation.first);
            (*beginCoeffsVec).setBestCoeffs(std::move(pairCoeffsEvaluation.second));
            if (unlikely(verbose))
                --(*leftTasks);                
        }
    }

    bool GmdhModel::nextLevelCondition(int kBest, uint8_t pAverage, VectorC& combinations,
                                  const Criterion& criterion, SplittedData& data, double limit) {

        decltype(auto) model = this;
        auto func = [model = model](const MatrixXd& x, const VectorU16& comb) {return model->xDataForCombination(x, comb); };
        auto _bestCombinations{ criterion.getBestCombinations(combinations, data, func, kBest) };
        currentLevelEvaluation = getMeanCriterionValue(_bestCombinations, pAverage);

        if ((lastLevelEvaluation - currentLevelEvaluation > limit)) {
            lastLevelEvaluation = currentLevelEvaluation;
            if (preparations(data, _bestCombinations)) {
                ++level;
                return true;
            }
        }

        removeExtraCombinations();
        return false;
    }

    GmdhModel& GmdhModel::fit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion, int kBest, 
                              double testSize, uint8_t pAverage, int threads, int verbose, double limit) {

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
        lastLevelEvaluation = std::numeric_limits<double>::max();

        MatrixXd modifiedX{ x.rows(), x.cols() + 1 };
        modifiedX.col(x.cols()).setOnes();
        modifiedX.leftCols(x.cols()) = x;
        auto data{ splitData(modifiedX, y, testSize) };
        modifiedX.resize(0, 0); // TODO: clear???

        /*std::cout << data.xTrain << "\n\n";
        std::cout << data.xTest << "\n\n";
        std::cout << data.yTrain << "\n\n";
        std::cout << data.yTest << "\n\n";*/
        bool goToTheNextLevel;
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
                    option::BarWidth{ 25 },
                    option::Start{ "LEVEL " + std::to_string(level) + ((std::to_string(level).size() == 1) ? "  [" : " [")},
                    option::End{ "]" },
                    option::Lead{ ">" },
                    option::ShowElapsedTime{ true },
                    option::ShowPercentage{ true },
                    option::PostfixText{ "(" + std::to_string(evaluationCoeffsVec.size()) + " combinations)" }
                );
                show_console_cursor(false); // TODO: wtf???
                progressBar->set_progress(0);
            }
            decltype(auto) model = this;
            auto combsPortion{ static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads))) };
            for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) {
                boost::packaged_task<void> pt([model = static_cast<const GmdhModel*>(model),
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
                    #ifdef GMDH_MODULE
if (PyErr_CheckSignals() != 0) {
    pool.stop();
            pool.join();
    //boost::when_all(std::begin(futures), std::end(futures)).get();  
            return *this;
                //throw pybind11::error_already_set();
        }
            #endif
            
                    if (progressBar->current() < 100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size())
                        progressBar->set_progress(100.0 * (evaluationCoeffsVec.size() - leftTasks) / evaluationCoeffsVec.size());
                    boost::this_thread::sleep_for(boost::chrono::milliseconds(20));
                }
            }
            else {
                boost::when_all(std::begin(futures), std::end(futures)).get();   
                #ifdef GMDH_MODULE
if (PyErr_CheckSignals() != 0) {

            return *this;
        }
            #endif
            } 
            goToTheNextLevel = nextLevelCondition(kBest, pAverage, evaluationCoeffsVec, criterion, data, limit);

            if (verbose)
            {
                std::string stringError;
                std::ostringstream stream;
                if (currentLevelEvaluation == 0)
                    stringError = "0";
                else {
                    if (currentLevelEvaluation >= 1e-6 && currentLevelEvaluation < 1e6)
                        stream << std::fixed << std::setprecision(6);
                    stream << currentLevelEvaluation;
                    stringError = stream.str();
                    boost::trim_right_if(stringError, boost::is_any_of("0"));
                    boost::trim_right_if(stringError, boost::is_any_of("."));
                }
                progressBar->set_option(option::PostfixText("(" + std::to_string(evaluationCoeffsVec.size()) + " combinations) error=" + stringError));
                progressBar->set_progress(100);
            }
        } while (goToTheNextLevel);
        if (verbose)
            show_console_cursor(true);
        return *this;   
    }

    int validateInputData(double* testSize, uint8_t* pAverage, int* threads, int* kBest) {
        auto errorCode{ 0 };
#ifdef GMDH_LIB
        std::cout << DISPLAYEDCOLORWARNING;
#elif GMDH_MODULE
        //auto warnings = pybind11::module::import("warnings");
        auto sys = pybind11::module::import("sys");
#endif
        if (*testSize <= 0) { // TODO: add range 
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("value of testSize", "testSize = 0.5");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("value of test_size", "test_size = 0.5"), 1);
#endif
            * testSize = 0.5;
            errorCode |= 1;
        }
        if (threads && (*threads < -1 || !*threads))
        {
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("number of threads", "threads = 1");
#elif GMDH_MODULE
            //warnings.attr("warn")("old_foo() is deprecated, use new_foo() instead.");                
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("number of n_jobs", "n_jobs = 1"), 1);
#endif
            * threads = 1;
            errorCode |= 2;
        }
        if (pAverage && !(*pAverage)) {
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("number of pAverage", "pAverage = 1");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("number of p_average", "p_average = 1"), 1);
#endif
            * pAverage = 1;
            errorCode |= 4;
        }
        if (kBest && !(*kBest)) {
#ifdef GMDH_LIB
            std::cout << DISPLAYEDWARNINGMSG("number of kBest", "kBest = 1");
#elif GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, DISPLAYEDWARNINGMSG("number of k_best", "k_best = 1"), 1);
#endif
            * kBest = 1;
            errorCode |= 8;
        }
#ifdef GMDH_LIB
        std::cout << DISPLAYEDCOLORINFO;
#elif GMDH_MODULE
        sys.attr("stderr").attr("flush")();
#endif
        return errorCode;
    }

    std::string GmdhModel::getPolynomialCoeff(double coeff, int coeffIndex) const {
        std::string stringCoeff;
        if (coeff == 0)
            stringCoeff = "0";
        else {
            std::ostringstream stream;
            if (abs(coeff) >= 1e-4 && abs(coeff) < 1e6)
                stream << std::fixed << std::setprecision(4);
            stream << abs(coeff);
            stringCoeff = stream.str();
            boost::trim_right_if(stringCoeff, boost::is_any_of("0"));
            boost::trim_right_if(stringCoeff, boost::is_any_of("."));
        }
        return ((coeff >= 0) ? ((coeffIndex > 0) ? " + " : " ") : " - ") + stringCoeff;
    }

    int GmdhModel::save(const std::string& path) const {
        std::ofstream modelFile(path);
        if (!modelFile.is_open())
            return -1; // TODO: throw exception for Python
        else {
            modelFile << getModelName() << "\n" << inputColsNumber << "\n";
            for (int i = 0; i < bestCombinations.size(); ++i) {
                modelFile << "~\n";
                for (int j = 0; j < bestCombinations[i].size(); ++j)
                    modelFile << bestCombinations[i][j].getInfoForSaving();
            }
            modelFile.close();
        }
        return 0;
    }

    int GmdhModel::load(const std::string& path) {
        inputColsNumber = 0;
        bestCombinations.clear();

        std::ifstream modelFile(path);
        if (!modelFile.is_open())
            return -1; // TODO: throw exception for Python
        else {
            std::string modelName;
            modelFile >> modelName;
            if (modelName != getModelName())
                return -2; // TODO: throw exception for Python
            else {
                (modelFile >> inputColsNumber).get();

                int currLevel = -1;
                while (modelFile.peek() != EOF) {

                    if (modelFile.peek() == '~') {
                        std::string buffer;
                        std::getline(modelFile, buffer);
                        ++currLevel;
                        bestCombinations.push_back(VectorC());
                    }

                    std::string colsIndexesLine;
                    VectorU16 bestColsIndexes;
                    std::getline(modelFile, colsIndexesLine);
                    std::stringstream indexStream{ colsIndexesLine };
                    uint16_t index;
                    while (indexStream >> index)
                        bestColsIndexes.push_back(index);

                    std::string coeffsLine;
                    std::vector<double> coeffs;
                    std::getline(modelFile, coeffsLine);
                    std::stringstream coeffsStream{ coeffsLine };
                    double coeff;
                    while (coeffsStream >> coeff)
                        coeffs.push_back(coeff);

                    bestCombinations[currLevel].push_back({ std::move(bestColsIndexes), 
                                                            Map<VectorXd>(coeffs.data(), coeffs.size()) });
                }
            }
            modelFile.close();
        }
        return 0;
    }

    VectorXd GmdhModel::predict(const RowVectorXd& x, int lags) const {
        RowVectorXd expandedX(RowVectorXd::Zero(x.size() + lags));
        expandedX.leftCols(x.size()) = x;
        for (int i = 0; i < lags; ++i)
            expandedX(x.size() + i) = predict(expandedX(seq(i, x.size() + i - 1)))[0];
        return expandedX.rightCols(lags);
    }

    std::string GmdhModel::getBestPolynomial() const {
        std::string polynomialStr = "";
        for (int i = 0; i < bestCombinations.size(); ++i) {
            for (int j = 0; j < bestCombinations[i].size(); ++j) {
                auto bestColsIndexes = bestCombinations[i][j].combination();
                auto bestCoeffs = bestCombinations[i][j].bestCoeffs();
                polynomialStr += getPolynomialPrefix(i, j);
                for (int k = 0; k < bestCoeffs.size(); ++k) {
                    polynomialStr += getPolynomialCoeff(bestCoeffs[k], k);
                    polynomialStr += getPolynomialVariable(i, k, bestCoeffs.size(), bestColsIndexes);
                }
                if (i < bestCombinations.size() - 1 || j < bestCombinations[i].size() - 1)
                    polynomialStr += "\n";
            }
            if (i < bestCombinations.size() - 1 && bestCombinations[i].size() > 1)
                polynomialStr += "\n";
        }
        return polynomialStr;
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
