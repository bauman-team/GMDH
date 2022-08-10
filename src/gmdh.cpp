#include <iostream>
#include "gmdh.h"

namespace GMDH {

    std::string GMDH::getModelName() const
    {
        std::string modelName = std::string(boost::typeindex::type_id_runtime(*this).pretty_name());
        modelName = modelName.substr(modelName.find_last_of(':') + 1);
        return modelName;
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