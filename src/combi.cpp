﻿#include "combi.h"

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
        for (auto i : bestColsIndexes) modelFile << i << ' ';
        modelFile << "\n";
        for (auto i : bestCoeffs) modelFile << i << ' ';
        modelFile << "\n";
        modelFile.close();
    }
    return 0;
}

int COMBI::load(const std::string& path)
{
    inputColsNumber = 0;
    bestColsIndexes.clear();

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
            int index;
            while (indexStream >> index)
                bestColsIndexes.push_back(index);

            std::string coeffsLine;
            std::vector<double> coeffs;
            std::getline(modelFile, coeffsLine);
            std::stringstream coeffsStream(coeffsLine);
            double coeff;
            while (coeffsStream >> coeff)
                coeffs.push_back(coeff);
            bestCoeffs(coeffs);
        }
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
    return modifiedX(Eigen::all, bestColsIndexes) * bestCoeffs;
}

COMBI& COMBI::fit(MatrixXd x, VectorXd y, const Criterion& criterion, int threads, int verbose)
{   
    level = 1;
    threads = std::min(threads, (int)std::thread::hardware_concurrency());
    boost::asio::thread_pool pool(threads);
    boost::function<void(const MatrixXd&, const VectorXd&, const Criterion&, std::vector<std::vector<bool> >::const_iterator,
    std::vector<std::vector<bool> >::const_iterator, std::vector<std::pair<std::pair<double, VectorXd>, 
    std::vector<bool> >>::iterator)> calcEvaluationCoeffs = 
        [] (const MatrixXd& x, const VectorXd& y, const Criterion& criterion, std::vector<std::vector<bool> >::const_iterator beginComb, 
        std::vector<std::vector<bool> >::const_iterator endComb, std::vector<std::pair<std::pair<double, VectorXd>, 
        std::vector<bool> >>::iterator beginCoeffsVec) {
            for (; beginComb < endComb; ++beginComb, ++beginCoeffsVec) {
                std::vector<int> colsIndexes; // TODO: typedef (using) for all types
                for (int j = 0; j < beginComb->size(); ++j)
                    if ((*beginComb)[j])
                        colsIndexes.push_back(j);
                colsIndexes.push_back(x.cols() - 1);
                beginCoeffsVec->first = criterion.calculate(x(Eigen::all, colsIndexes), y);
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
              option::Start{"LEVEL " + std::to_string(level) + " ["},
              option::End{"]"},
              option::ShowElapsedTime{true},
              option::ShowPercentage{true},
              option::Lead{">"}
              //option::ForegroundColor{Color::white},
              //option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
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
                std::vector<int> colsIndexes;
                for (int j = 0; j < combinations[i].size(); ++j)
                    if (combinations[i][j])
                        colsIndexes.push_back(j);
                colsIndexes.push_back(x.cols());
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
            break; // TODO: fix bad code style
        }
        ++level;
    }

    bestColsIndexes = polynomialToIndexes(bestPolynomial);

    return *this;
}

std::string COMBI::getBestPolynomial() const
{
    std::string polynomialStr = "y =";
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

std::vector<std::vector<bool>> COMBI::getCombinations(int n_cols, int level) const
{
    std::vector<std::vector<bool>> combinations;
    std::vector<bool> combination(n_cols);
    std::fill(combination.begin(), combination.begin() + level, 1);
    do {
        combinations.push_back(combination);
    } while (std::prev_permutation(combination.begin(), combination.end()));
    return combinations;
}

std::vector<int> COMBI::polynomialToIndexes(const std::vector<bool>& polynomial) const
{
    std::vector<int> colsIndexes;
    for (int i = 0; i < polynomial.size(); ++i)
        if (polynomial[i])
            colsIndexes.push_back(i);
    colsIndexes.push_back(polynomial.size());
    return colsIndexes;
}
}