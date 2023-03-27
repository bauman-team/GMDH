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
        currLevelEvaluation += combBegin->evaluation();
    return currLevelEvaluation / static_cast<double>(k);
}

void GmdhModel::polynomialsEvaluation(const SplittedData& data, const Criterion& criterion,
    IterC beginCoeffsVec, IterC endCoeffsVec, std::atomic<int> *leftTasks, int verbose) const {
    for (; beginCoeffsVec < endCoeffsVec; ++beginCoeffsVec) {
        auto pairCoeffsEvaluation{ criterion.calculate(xDataForCombination(data.xTrain, beginCoeffsVec->combination()),
                                                        xDataForCombination(data.xTest, beginCoeffsVec->combination()),
                                                        data.yTrain, data.yTest) };
        beginCoeffsVec->setEvaluation(pairCoeffsEvaluation.first);
        beginCoeffsVec->setBestCoeffs(std::move(pairCoeffsEvaluation.second));
        if (unlikely(verbose > 0))
            --(*leftTasks);                
    }
}

bool GmdhModel::nextLevelCondition(int kBest, int pAverage, VectorC& combinations,
                                const Criterion& criterion, SplittedData& data, double limit) {

    decltype(auto) model = this;
    auto func = [model = model](const MatrixXd& x, const VectorU16& comb) {return model->xDataForCombination(x, comb); };
    auto _bestCombinations{ criterion.getBestCombinations(combinations, data, func, kBest) };
    currentLevelEvaluation = getMeanCriterionValue(_bestCombinations, pAverage);

    if ((lastLevelEvaluation - currentLevelEvaluation > limit) &&
        (lastLevelEvaluation = currentLevelEvaluation, preparations(data, std::move(_bestCombinations)))) {
        ++level;
        return true;
    }

    removeExtraCombinations();
    return false;
}

GmdhModel& GmdhModel::gmdhFit(const MatrixXd& x, const VectorXd& y, const Criterion& criterion,
            int kBest, double testSize, int pAverage, int threads, int verbose, double limit) {

    if (x.rows() != y.size())
        throw std::invalid_argument(getVariableName("x", "X") + " rows number and y size must be equal");

    using namespace indicators;
    using T = boost::packaged_task<void>;
    std::unique_ptr<ProgressBar> progressBar;

    boost::asio::thread_pool pool(threads); // reserving threads
    std::vector<boost::unique_future<T::result_type> > futures; // creating vector of futures on executable tasks
    futures.reserve(threads);
    std::atomic<int> leftTasks; // TODO: change to volatile structure

    level = 1; // reset last training
    inputColsNumber = x.cols();
    lastLevelEvaluation = std::numeric_limits<double>::max();

    auto data{ internalSplitData(x, y, testSize, true) };

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

        //evaluationCoeffsVec = VectorC{std::begin(combinations), std::end(combinations)}
        auto currLevelEvaluation{ std::begin(evaluationCoeffsVec) };
        for (auto it = std::begin(combinations); it != std::end(combinations); ++it, ++currLevelEvaluation)
            currLevelEvaluation->setCombination(std::move(*it));

        if (verbose > 0) {
            leftTasks = static_cast<int>(evaluationCoeffsVec.size()); // seting up counter for verbose
            progressBar = std::make_unique<ProgressBar>(
                option::BarWidth{ 25 },
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
        auto combsPortion{ static_cast<int>(std::ceil(evaluationCoeffsVec.size() / static_cast<double>(threads))) }; // spliting all combinations on portions for threads calculating
        for (auto i = 0; i * combsPortion < evaluationCoeffsVec.size(); ++i) {
            boost::packaged_task<void> pt([model = static_cast<const GmdhModel*>(model),
                &data = static_cast<const SplittedData&>(data), &criterion = static_cast<const Criterion&>(criterion), 
                &evaluationCoeffsVec, &leftTasks, verbose, combsPortion, i]() {
                    model->polynomialsEvaluation(data, criterion, std::begin(evaluationCoeffsVec) + combsPortion * i,
                        std::begin(evaluationCoeffsVec) + std::min(static_cast<size_t>(combsPortion * (i + 1)), 
                        evaluationCoeffsVec.size()), &leftTasks, verbose); });
            futures.push_back(pt.get_future()); // saving future on task
            post(pool, std::move(pt)); // starting task executions
        } 

        if (verbose > 0) {
            while (leftTasks) {
#ifdef GMDH_MODULE
                if (PyErr_CheckSignals() != 0) { // handling keyboard (ctrl+c) interruption
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
            boost::when_all(std::begin(futures), std::end(futures)).get(); // waiting until all tasks are completed
#ifdef GMDH_MODULE
            if (PyErr_CheckSignals() != 0) // handling keyboard (ctrl+c) interruption
                return *this;
#endif
        } 
        goToTheNextLevel = nextLevelCondition(kBest, pAverage, evaluationCoeffsVec, criterion, data, limit); // checking the results of the current level for improvement

        if (verbose > 0)
        {
            std::string stringError;
            std::ostringstream stream;
            if (currentLevelEvaluation == 0)
                stringError = "0"; // LCOV_EXCL_LINE
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
    if (verbose > 0)
        show_console_cursor(true);
    return *this;   
}

boost::json::value GmdhModel::toJSON() const {
    return boost::json::object{
        {"modelName", getModelName()},
        {"inputColsNumber", inputColsNumber},
        {"bestCombinations", bestCombinations},
    };
} // LCOV_EXCL_LINE

int GmdhModel::fromJSON(boost::json::value jsonModel) { // TODO: maybe add try/catch
    auto& o = jsonModel.as_object();
    std::string modelName = o.at("modelName").as_string().c_str();
    if (modelName != getModelName()) // checking for compliance with the current model and the saved one
        return 3; // TODO: add defines for error numbers
    bestCombinations.clear();
    inputColsNumber = o.at("inputColsNumber").as_int64();
    auto& bestCombs = o.at("bestCombinations").as_array();
    for (auto vc : bestCombs) {
        bestCombinations.push_back(VectorC());
        auto& bestComb = vc.as_array();
        for (auto c : bestComb)
            bestCombinations.back().push_back(boost::json::value_to<Combination>(c));
    }
    return 0;
}

int validateInputData(double* testSize, int* pAverage, int* threads, int* verbose, double* limit, int* kBest) {
    auto errorCode{ 0 };
#ifdef GMDH_MODULE
    auto sys = pybind11::module::import("sys");
#else
    std::cout << DISPLAYEDCOLORWARNING;
#endif
    // block with exceptions 
    if (*testSize <= 0 || *testSize >= 1) { // TODO: add range 
        std::string errorMsg = getVariableName("testSize", "test_size") + " value must be in the (0, 1) range";
        throw std::invalid_argument(errorMsg);
        //errorCode |= 1;
    }
    if (pAverage && *pAverage < 1) {
        std::string errorMsg = getVariableName("pAverage", "p_average") + " value must be a positive integer";
        throw std::invalid_argument(errorMsg);
        //errorCode |= 4;
    }
    if (limit && *limit < 0) {
        std::string errorMsg = getVariableName("limit", "limit") + " value must be non-negative";
        throw std::invalid_argument(errorMsg);
    }
    if (kBest && *kBest < 1) {
        std::string errorMsg = getVariableName("kBest", "k_best") + " value must be a positive integer";
        throw std::invalid_argument(errorMsg);
        //errorCode |= 8;
    }

    // block with warnings 
    if (threads)
    {
        if (*threads == -1)
            *threads = boost::thread::hardware_concurrency(); // TODO: maybe find optimal count based on data.size() and hardware_concurrency()
        else if (*threads < 1 || *threads > boost::thread::hardware_concurrency()) {
            if (*threads < 1) {
#ifdef GMDH_MODULE
                PyErr_WarnEx(PyExc_Warning, MINTHREADSWARNING("n_jobs"), 1);
#else
                std::cout << MINTHREADSWARNING("threads");
#endif
                *threads = 1;
            }  
            else if (*threads > boost::thread::hardware_concurrency()) {
#ifdef GMDH_MODULE
                PyErr_WarnEx(PyExc_Warning, MAXTHREADSWARNING("n_jobs"), 1);
#else
                std::cout << MAXTHREADSWARNING("threads");
#endif
                *threads = boost::thread::hardware_concurrency(); // TODO: change limit
            }
            //errorCode |= 2;
        }
    }
    if (verbose) {
        if (*verbose < 0) {
#ifdef GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, MINVERBOSEWARNING("verbose"), 1);
#else
            std::cout << MINVERBOSEWARNING("verbose");
#endif
            *verbose = 0;
        }
        else if (*verbose > MAXVERBOSENUMBER) {
#ifdef GMDH_MODULE
            PyErr_WarnEx(PyExc_Warning, MAXVERBOSEWARNING("verbose"), 1);
#else
            std::cout << MAXVERBOSEWARNING("verbose");
#endif
            *verbose = MAXVERBOSENUMBER;
        }
    }

#ifdef GMDH_MODULE
    sys.attr("stderr").attr("flush")();
#else
    std::cout << DISPLAYEDCOLORINFO;
#endif
    return errorCode;
}

std::string GmdhModel::getPolynomialCoeffSign(double coeff, bool isFirstCoeff) const {
    return ((coeff >= 0) ? ((isFirstCoeff) ? " " : " + ") : " - ");
}

std::string GmdhModel::getPolynomialCoeffValue(double coeff, bool isLastCoeff) const {
    std::string stringCoeff;
    std::ostringstream stream;
    if (abs(coeff) >= 1e-4 && abs(coeff) < 1e6)
        stream << std::fixed << std::setprecision(4);
    stream << abs(coeff);
    stringCoeff = stream.str();
    boost::trim_right_if(stringCoeff, boost::is_any_of("0"));
    boost::trim_right_if(stringCoeff, boost::is_any_of("."));
    return ((stringCoeff != "1" || isLastCoeff) ? stringCoeff : "");
}

int GmdhModel::save(const std::string& path) const {
    std::ofstream modelFile(path);
    if (!modelFile.is_open())
#ifdef GMDH_MODULE
        throw FileException("Invalid argument: '" + path + "'"); 
#else
        return 1; 
#endif
    else {
        modelFile << toJSON();
        modelFile.close();
    }
    return 0;
}

int GmdhModel::load(const std::string& path) {        
    if (!boost::filesystem::is_regular_file(path)) // TODO: maybe remove because extra checking
#ifdef GMDH_MODULE
        throw FileException("Invalid argument: '" + path + "'");
#else
        return 1; 
#endif
    else {
        std::ifstream modelFile(path);
        std::string inputJSON(std::istreambuf_iterator<char>(modelFile), {});
            
        boost::json::error_code ec;
        auto jsonValue = boost::json::parse(inputJSON, ec);
        if (ec) {
            modelFile.close();
#ifdef GMDH_MODULE
            throw FileException(CORRUPTEDFILEEXCEPTION);
#else
            return 2; 
#endif
        }

        auto errorCode = fromJSON(jsonValue); 
        if (errorCode) {
            modelFile.close();
#ifdef GMDH_MODULE
            std::string inputModel{ jsonValue.as_object().at("modelName").as_string().c_str() };
            throw FileException(WRONGMODELFILEEXCEPTION(inputModel, getModelName()));
#else
            return errorCode; 
#endif
        }

        modelFile.close();
    }
    return 0;
}

VectorXd GmdhModel::predict(const RowVectorXd& x, int lags) const {
    if (lags <= 0) {
        std::string errorMsg = "lags value must be a positive integer";
        throw std::invalid_argument(errorMsg);
    }
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
            bool isFirstCoeff = true;
            for (int k = 0; k < bestCoeffs.size(); ++k) {
                if (bestCoeffs[k]) {
                    polynomialStr += getPolynomialCoeffSign(bestCoeffs[k], isFirstCoeff);
                    auto coeffValuelStr = getPolynomialCoeffValue(bestCoeffs[k], k == bestCoeffs.size() - 1);
                    polynomialStr += coeffValuelStr;
                    if (coeffValuelStr != "" && k != bestCoeffs.size() - 1)
                        polynomialStr += "*";
                    polynomialStr += getPolynomialVariable(i, k, bestCoeffs.size(), bestColsIndexes);
                    isFirstCoeff = false;
                }
            }
            if (i < bestCombinations.size() - 1 || j < bestCombinations[i].size() - 1)
                polynomialStr += "\n";
        }
        if (i < bestCombinations.size() - 1 && bestCombinations[i].size() > 1)
            polynomialStr += "\n";
    }
    return polynomialStr;
}

std::string&& getVariableName(std::string&& cppName, std::string&& pyName) {
#ifdef GMDH_MODULE
    return std::move(pyName);
#else
    return std::move(cppName);
#endif
}

PairMVXd timeSeriesTransformation(const VectorXd& timeSeries, int lags) {
    std::string errorMsg = "";
    if (timeSeries.size() == 0)
        errorMsg = getVariableName("timeSeries", "time_series") + " value is empty";
    else if (lags <= 0)
        errorMsg = "lags value must be a positive integer";
    else if (lags >= timeSeries.size())
        errorMsg = "lags value can't be greater than " + getVariableName("timeSeries", "time_series") + " size";
    if (errorMsg != "")
        throw std::invalid_argument(errorMsg);
            
    VectorXd yTimeSeries{ timeSeries.tail(timeSeries.size() - lags) };
    MatrixXd xTimeSeries{ timeSeries.size() - lags, lags };
    for (auto i = 0; i < timeSeries.size() - lags; ++i)
        xTimeSeries.row(i) = timeSeries.segment(i, lags);
    return { std::move(xTimeSeries), std::move(yTimeSeries) };
}

SplittedData GmdhModel::internalSplitData(const MatrixXd& x, const VectorXd& y, double testSize, bool addOnesCol) {
    SplittedData data;
    int testItemsNumber = round(x.rows() * testSize);

    if (addOnesCol) {
        data.xTrain.resize(x.rows() - testItemsNumber, x.cols() + 1);
        data.xTest.resize(testItemsNumber, x.cols() + 1);

        data.xTrain.leftCols(x.cols()) = x.topRows(x.rows() - testItemsNumber);
        data.xTrain.col(x.cols()).setOnes();

        data.xTest.leftCols(x.cols()) = x.bottomRows(testItemsNumber);
        data.xTest.col(x.cols()).setOnes();
    }
    else {
        data.xTrain = x.topRows(x.rows() - testItemsNumber);
        data.xTest = x.bottomRows(testItemsNumber);
    }

    data.yTrain = y.head(y.size() - testItemsNumber);
    data.yTest = y.tail(testItemsNumber);

    return data;
}

void GmdhModel::checkMatrixColsNumber(const MatrixXd& x) const {
    if (inputColsNumber != x.cols()) {
        std::string varName = getVariableName("x", "X");
        std::string needCols = std::to_string(inputColsNumber);
        std::string errorMsg =  "Matrix '" + varName + "' must have " + needCols + 
            " columns because there were " + needCols + " columns in the training '" + varName + "' matrix";
        throw std::invalid_argument(errorMsg);
    }
}

SplittedData splitData(const MatrixXd& x, const VectorXd& y, double testSize, bool shuffle, int randomSeed) {
    validateInputData(&testSize);
    std::string errorMsg = "";
    if (x.rows() != y.size())
        errorMsg = getVariableName("x", "X") + " rows number and y size must be equal";
    else if (round(x.rows() * testSize) == 0 || round(x.rows() * testSize) == x.rows())
        errorMsg = "Result contains an empty array. Change the arrays size or the " + 
            getVariableName("testSize", "test_size") + " value for correct splitting";
    if (errorMsg != "")
        throw std::invalid_argument(errorMsg);

    SplittedData data;
    if (!shuffle)
        data = GmdhModel::internalSplitData(x, y, testSize);
    else {
        std::mt19937_64 randGen;
        if (randomSeed == 0)  randomSeed = std::chrono::system_clock::now().time_since_epoch().count();
        randGen.seed(randomSeed);

        VectorI shuffled_rows_indexes(x.rows());
        std::iota(std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes), 0);
        std::shuffle(std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes), randGen);

        int testItemsNumber = round(x.rows() * testSize);

        VectorI train_indexes{ std::begin(shuffled_rows_indexes), std::end(shuffled_rows_indexes) - testItemsNumber };
        VectorI test_indexes{ std::end(shuffled_rows_indexes) - testItemsNumber, std::end(shuffled_rows_indexes) };

        data.xTrain = x(train_indexes, Eigen::all);
        data.xTest = x(test_indexes, Eigen::all);
        data.yTrain = y(train_indexes);
        data.yTest = y(test_indexes);
    }
    return data;
}
}
