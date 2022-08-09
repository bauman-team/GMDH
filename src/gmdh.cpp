#include <iostream>
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

    std::string GMDH::getModelName() const
    {
        std::string model_name = std::string(boost::typeindex::type_id_with_cvr<decltype(this)>().name());
        model_name = model_name.substr(6, model_name.find(' ', 6) - 6);
        return model_name;
    }

    GMDH::GMDH()
    {
        level = 1;
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
}