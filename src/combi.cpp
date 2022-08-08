#include "combi.h"
#define THREADS_USING
#define THREADS_COUNT 8

namespace GMDH {

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

}