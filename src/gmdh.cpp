#include "gmdh.h"

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

    void COMBI::save() const
    {
        return;
    }

    int COMBI::load()
    {
        return 0;
    }

    double COMBI::predict(rowvec x) const
    {
        return predict(mat(x))(0);
    }

    vec COMBI::predict(mat x) const
    {
        x.insert_cols(x.n_cols, vec(x.n_rows, fill::ones));
        return x.cols(best_cols_index) * best_coeffs;
    }

    COMBI& COMBI::fit(mat x, vec y, const Criterion& criterion)
    {
        //std::unordered_multimap<double, std::vector<bool>>
        // TODO: add using (as typedef)
        double last_level_evaluation = std::numeric_limits<double>::max();

        x.insert_cols(x.n_cols, vec(x.n_rows, fill::ones));

        while (level < x.n_cols)
        {
            std::vector<std::pair<std::pair<double, vec>, std::vector<bool> >> evaluation_coeffs_vec; // TODO: add reserve
            std::vector<std::pair<std::pair<double, vec>, std::vector<bool> >>::const_iterator curr_level_evaluation;
            std::vector<std::vector<bool>> combinations = getCombinations(x.n_cols - 1, level);
            for (int i = 0; i < combinations.size(); ++i)
            {
                std::vector<u64> cols_index;
                for (int j = 0; j < combinations[i].size(); ++j)
                    if (combinations[i][j])
                        cols_index.push_back(j);
                cols_index.push_back(x.n_cols - 1);
                mat comb_x = x.cols(uvec(cols_index));
                evaluation_coeffs_vec.push_back(std::pair<std::pair<double, vec>, std::vector<bool> >(criterion.calculate(comb_x, y), combinations[i]));
            }
            if (last_level_evaluation > 
            (curr_level_evaluation = std::min_element(
            std::cbegin(evaluation_coeffs_vec), // TODO: maybe begin() for move value
            std::cend(evaluation_coeffs_vec), 
            [](std::pair<std::pair<double, vec>, std::vector<bool> > first, 
            std::pair<std::pair<double, vec>, std::vector<bool> > second) { 
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

        std::vector<u64> cols_index;
        for (int i = 0; i < best_polinom.size(); ++i)
            if (best_polinom[i])
                cols_index.push_back(i);
        cols_index.push_back(best_polinom.size());
        best_cols_index = cols_index;

        return *this;
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

    RegularityCriterion::RegularityCriterion(double _test_size, bool _shuffle, int _random_seed) : RegularityCriterionTS(_test_size)
    {
        shuffle = _shuffle;
        random_seed = _random_seed;
    }

    vec Criterion::internalCriterion(mat x_train, vec y_train) const
    {
        return solve(x_train, y_train);
    }

    std::pair<double, vec> RegularityCriterion::calculate(mat x, vec y) const
    {
        if (!shuffle)
            return RegularityCriterionTS::calculate(x, y);

        if (random_seed != 0)
            arma::arma_rng::set_seed(random_seed);
        else
            arma::arma_rng::set_seed_random();

        uvec shuffled_rows_indexes = randperm(x.n_rows);
        uvec train_indexes = shuffled_rows_indexes.head(x.n_rows - round(x.n_rows * test_size));
        uvec test_indexes = shuffled_rows_indexes.tail(round(x.n_rows * test_size));

        mat x_train = x.rows(train_indexes);
        mat x_test = x.rows(test_indexes);
        vec y_train = y.elem(train_indexes);
        vec y_test = y.elem(test_indexes);

        return getCriterionValue(x_train, y_train, x_test, y_test);
    }

    mat polynomailFeatures(const mat X, int max_degree) {
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
    } 

    std::pair<double, vec> RegularityCriterionTS::getCriterionValue(mat x_train, vec y_train, mat x_test, vec y_test) const
    {
        //vec coeffs = internalCriterion(x_train, y_train);
        vec coeffs(x_test.n_cols, fill::randu);
        vec y_pred = x_test * coeffs;
        return std::pair<double, vec>(sum(square(y_test - y_pred)) / sum(square(y_test)), coeffs);
    }

    RegularityCriterionTS::RegularityCriterionTS(double _test_size)
    {
        if (_test_size > 0 && _test_size < 1)
            test_size = _test_size;
        else
            throw; // TODO: exception???
    }

    std::pair<double, vec> RegularityCriterionTS::calculate(mat x, vec y) const
    {
        mat x_train = x.head_rows(x.n_rows - round(x.n_rows * test_size));
        mat x_test = x.tail_rows(round(x.n_rows * test_size));
        vec y_train = y.head(y.n_elem - round(y.n_elem * test_size));
        vec y_test = y.tail(round(y.n_elem * test_size));

        return getCriterionValue(x_train, y_train, x_test, y_test);
    }
}