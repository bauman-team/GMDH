#include "gmdh.h"
#include <iostream>

#define THREADS_USING
#define THREADS_COUNT 8

namespace GMDH {

    /*unsigned long COMBI::nChoosek(unsigned long n, unsigned long k)
    {
        if (k > n) return 0;
        if (k * 2 > n) k = n - k;
        if (k == 0) return 1;

        unsigned long result = n;
        for(unsigned long i = 2; i <= k; ++i) {
            result /= i;
            result *= (n - i + 1);
        }
        return result;
    }*/

    GMDH::GMDH()
    {
        level = 1;
    }

    COMBI::COMBI()
    {
        model_name = "COMBI";
    }

    int COMBI::save(std::string path) const
    {
        int status = 0;
        std::ofstream model_file;
        model_file.open(path);
        if (!model_file.is_open())
            status = -1;
        else
        {
            model_file << model_name << "\n";
            model_file << input_cols_number << "\n";
            for (auto i : best_cols_index) model_file << i << ' ';
            model_file << "\n";
            for (auto i : best_coeffs) model_file << i << ' ';
            model_file << "\n";
            model_file.close();
        }
        return status;
    }

    int COMBI::load(const std::string& path)
    {
        input_cols_number = 0;
        best_cols_index.clear();

        std::ifstream model_file;
        model_file.open(path);
        if (!model_file.is_open())
            return -1;
        else
        {
            std::string model_name;
            model_file >> model_name;
            if (model_name != model_name)
                return -1;
            else
            {
                (model_file >> input_cols_number).get();

                std::string cols_index_line;
                std::getline(model_file, cols_index_line);
                std::stringstream index_stream(cols_index_line);
                int index;
                while (index_stream >> index)
                    best_cols_index.push_back(index);

                std::string coeffs_line;
                std::vector<double> coeffs;
                std::getline(model_file, coeffs_line);
                std::stringstream coeffs_stream(coeffs_line);
                double coeff;
                while (coeffs_stream >> coeff)
                    coeffs.push_back(coeff);
                best_coeffs(coeffs);
            }
        }
        return 0;
    }

    double COMBI::predict(RowVectorXd x) const
    {
        return predict(MatrixXd(x))[0];
    }

    VectorXd COMBI::predict(MatrixXd x) const
    {
        MatrixXd xx(x.rows(), x.cols() + 1);
        xx.col(x.cols()).setOnes();
        xx.leftCols(x.cols()) = x;
        return xx(Eigen::all, best_cols_index) * best_coeffs;
    }

    COMBI& COMBI::fit(MatrixXd x, VectorXd y, const Criterion& criterion)
    {        
        boost::asio::thread_pool pool(THREADS_COUNT); // TODO: variable for count of threads
        boost::function<void(const MatrixXd&, const VectorXd&, const Criterion&, std::vector<std::vector<bool> >::const_iterator, 
        std::vector<std::vector<bool> >::const_iterator, std::vector<std::pair<std::pair<double, VectorXd>, 
        std::vector<bool> >>::iterator)> calc_evaluation_coeffs = 
            [] (const MatrixXd& x, const VectorXd& y, const Criterion& criterion, std::vector<std::vector<bool> >::const_iterator begin_comb, 
            std::vector<std::vector<bool> >::const_iterator end_comb, std::vector<std::pair<std::pair<double, VectorXd>, 
            std::vector<bool> >>::iterator begin_coeff_vec) {
                for (; begin_comb < end_comb; ++begin_comb, ++begin_coeff_vec) {
                    std::vector<int> cols_index; // TODO: typedef (using) for all types
                    for (int j = 0; j < begin_comb->size(); ++j)
                        if ((*begin_comb)[j])
                            cols_index.push_back(j);
                    cols_index.push_back(x.cols() - 1);
                    MatrixXd comb_x = x(Eigen::all, cols_index);

                    begin_coeff_vec->first = criterion.calculate(comb_x, y);
                    begin_coeff_vec->second = *begin_comb;
                }      
            };

        double last_level_evaluation = std::numeric_limits<double>::max();
        std::vector<bool> best_polinom;
        input_cols_number = x.cols();

        MatrixXd xx(x.rows(), x.cols() + 1);
        xx.col(x.cols()).setOnes();
        xx.leftCols(x.cols()) = x;

        while (level < xx.cols()) {
            std::vector<std::pair<std::pair<double, VectorXd>, std::vector<bool> >> evaluation_coeffs_vec; 
            std::vector<std::pair<std::pair<double, VectorXd>, std::vector<bool> >>::const_iterator curr_level_evaluation; // TODO: add using (as typedef)
            std::vector<std::vector<bool> > combinations = getCombinations(x.cols(), level);
#ifdef THREADS_USING
            using T = boost::packaged_task<void>;
            std::vector<boost::unique_future<T::result_type> > futures; // TODO: reserve??? or array
            
            evaluation_coeffs_vec.resize(combinations.size());
            auto count_thread_comb = static_cast<int>(std::ceil(combinations.size() / static_cast<double>(THREADS_COUNT))); 
            for (auto i = 0; i * count_thread_comb < combinations.size(); ++i) {
                boost::packaged_task<void> pt(boost::bind(calc_evaluation_coeffs, xx, y, boost::ref(criterion),
                combinations.cbegin() + count_thread_comb * i, 
                combinations.cbegin() + std::min(static_cast<size_t>(count_thread_comb * (i + 1)), combinations.size()),
                evaluation_coeffs_vec.begin() + count_thread_comb * i));
                futures.push_back(pt.get_future());
                post(pool, std::move(pt));
            }
            boost::when_all(futures.begin(), futures.end()).get();
#else
            evaluation_coeffs_vec.reserve(combinations.size());
            for (int i = 0; i < combinations.size(); ++i)
            {
                std::vector<int> cols_index;
                for (int j = 0; j < combinations[i].size(); ++j)
                    if (combinations[i][j])
                        cols_index.push_back(j);
                cols_index.push_back(x.cols());
                MatrixXd comb_x = xx(Eigen::all, cols_index); 
                evaluation_coeffs_vec.push_back(std::pair<std::pair<double, VectorXd>, std::vector<bool> >(criterion.calculate(comb_x, y), combinations[i]));
            }
#endif
            // > or >= ?
            if (last_level_evaluation > 
            (curr_level_evaluation = std::min_element(
            std::cbegin(evaluation_coeffs_vec), 
            std::cend(evaluation_coeffs_vec), 
            [](std::pair<std::pair<double, VectorXd>, std::vector<bool> > first, 
            std::pair<std::pair<double, VectorXd>, std::vector<bool> > second) { 
                return first.first.first < second.first.first;
            }))->first.first) {
                last_level_evaluation = curr_level_evaluation->first.first;
                best_polinom = curr_level_evaluation->second;
                best_coeffs = curr_level_evaluation->first.second;
            }
            else {
                break; // TODO: fix bad code style
            }
            ++level;
        }

        best_cols_index = polinomToIndexes(best_polinom);

        return *this;
    }

    std::string COMBI::getBestPolymon() const
    {
        std::string polynom_str = "y =";
        for (int i = 0; i < best_cols_index.size(); ++i)
        {
            if (best_coeffs[i] > 0)
            {
                if (i > 0)
                    polynom_str += " + ";
                else
                    polynom_str += " ";
            }
            else
                polynom_str += " - ";
            polynom_str += std::to_string(abs(best_coeffs[i]));
            if (i != best_cols_index.size() - 1)
                polynom_str += "*x" + std::to_string(best_cols_index[i] + 1);
        }
        return polynom_str;
    }

    std::vector<std::vector<bool>> COMBI::getCombinations(int n, int k) const
    {
        std::vector<std::vector<bool>> combinations;
        std::vector<bool> combination(n);
        std::fill(combination.begin(), combination.begin() + k, 1);
        do {
            combinations.push_back(combination);
        } while (std::prev_permutation(combination.begin(), combination.end()));
        return combinations;
    }

    std::vector<int> COMBI::polinomToIndexes(std::vector<bool> polinom) const
    {
        std::vector<int> cols_index;
        for (int i = 0; i < polinom.size(); ++i)
            if (polinom[i])
                cols_index.push_back(i);
        cols_index.push_back(polinom.size());
        return cols_index;
    }

    RegularityCriterion::RegularityCriterion(double _test_size, bool _shuffle, int _random_seed) : RegularityCriterionTS(_test_size)
    {
        shuffle = _shuffle;
        random_seed = _random_seed;
    }

    VectorXd Criterion::internalCriterion(MatrixXd x_train, VectorXd y_train) const
    { 
        return x_train.colPivHouseholderQr().solve(y_train); // TODO: add parameter to choose solve algorithm
    }

    std::pair<double, VectorXd> RegularityCriterion::calculate(const MatrixXd& x, const VectorXd& y) const
    {
        return getCriterionValue(splitData(x, y, test_size, shuffle, random_seed));
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
        VectorXd y_ts = x.tail(x.size() - lags);
        MatrixXd x_ts(x.size() - lags, lags);
        for (int i = 0; i < x.size() - lags; ++i) {
            x_ts.row(i) = x.segment(i, lags);
        }
        return std::pair<MatrixXd, VectorXd>(x_ts, y_ts);
    }

    splitted_data splitTsData(MatrixXd x, VectorXd y, double test_size)
    {
        splitted_data data;
        data.x_train = x.topRows(x.rows() - round(x.rows() * test_size));
        data.x_test = x.bottomRows(round(x.rows() * test_size));
        data.y_train = y.head(y.size() - round(y.size() * test_size));
        data.y_test = y.tail(round(y.size() * test_size));
        return data;
    }

    splitted_data splitData(MatrixXd x, VectorXd y, double test_size, bool shuffle, int _random_seed)
    {
        if (!shuffle)
            return splitTsData(x, y, test_size);

        if (_random_seed != 0)
            std::srand(_random_seed);
        else
            std::srand(std::time(NULL));

        
        std::vector<int> shuffled_rows_indexes(x.rows());
        std::iota(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end(), 0);
        std::random_shuffle(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end());

        std::vector<int> train_indexes(shuffled_rows_indexes.begin(), shuffled_rows_indexes.end() - round(x.rows() * test_size));
        std::vector<int> test_indexes(shuffled_rows_indexes.end() - round(x.rows() * test_size), shuffled_rows_indexes.end());
        
        splitted_data data;
        data.x_train = x(train_indexes, Eigen::all);
        data.x_test = x(test_indexes, Eigen::all);
        data.y_train = y(train_indexes);
        data.y_test = y(test_indexes);

        return data;
    }

    std::pair<double, VectorXd> RegularityCriterionTS::getCriterionValue(splitted_data data) const
    {
        VectorXd coeffs = internalCriterion(data.x_train, data.y_train);
        //vec coeffs(data.x_test.n_cols, fill::randn);
        VectorXd y_pred = data.x_test * coeffs;
        return std::pair<double, VectorXd>(((data.y_test - y_pred).array().square().sum() / data.y_test.array().square().sum()), coeffs);
    }

    RegularityCriterionTS::RegularityCriterionTS(double _test_size)
    {
        if (_test_size > 0 && _test_size < 1)
            test_size = _test_size;
        else
            throw; // TODO: exception???
    }

    std::pair<double, VectorXd> RegularityCriterionTS::calculate(const MatrixXd& x, const VectorXd& y) const
    {
        return getCriterionValue(splitTsData(x, y, test_size));
    }
}