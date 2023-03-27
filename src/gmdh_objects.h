namespace GMDH {

using namespace Eigen;

/// @brief A type definition for storage of the indexes of data columns used in the model
using VectorU16 = std::vector<uint16_t>;

/// @brief A type definition for iterator to the storage of indexes of the data columns used in the model
using IterU16 = VectorU16::iterator;

/// @brief A type definition for storages vector of the indexes of data columns used in the models
using VectorVu16 = std::vector<VectorU16>;

/// @brief A type definition for storing a pair of values of external criterion and corresponding model coefficients
using PairDVXd = std::pair<double, VectorXd>;

/// @brief A type definition for storing a pair of X and y data of the transformed time series
using PairMVXd = std::pair<MatrixXd, VectorXd>;

/// @brief A type definition for storing data row indexes during the shuffing process
using VectorI = std::vector<int>;

class Criterion;

/// @brief Structure for storing parts of a split dataset
struct GMDH_API SplittedData {
    MatrixXd xTrain; ///< The first part of the input X matrix
    MatrixXd xTest; ///< The second part of the input X matrix
    VectorXd yTrain; ///< The first part of the input y vector
    VectorXd yTest; ///< The second part of the input y vector
};


/// @brief Сlass representing the candidate model of the GMDH algorithm
class GMDH_API Combination { 
    VectorU16 _combination; //!< Vector of the X matrix column indexes used to construct polynomial of the candidate model
    VectorXd _bestCoeffs; ///< Vector of the calculated coefficients corresponding to the polynomial variables of the candidate model
    double _evaluation; ///< Value of the external criterion evaluation for the candidate model
public:
    /// @brief Construct a new Combination object
    Combination() {}

    /**
     * @brief Construct a new Combination objects
     * 
     * @param comb Vector of the X matrix column indexes that should be used in the polynomial of the candidate model
     */
    Combination(VectorU16 comb) : _combination(comb) {} 

    /**
     * @brief Construct a new Combination object
     * 
     * @param comb Vector of the X matrix column indexes that should be used in the polynomial of the candidate model
     * @param coeffs Vector of the calculated coefficients corresponding to the polynomial variables of the candidate model
     */
    Combination(VectorU16&& comb, VectorXd&& coeffs) : _combination(std::move(comb)), _bestCoeffs(std::move(coeffs)) {} 

    /**
     * @brief Get the vector of the X matrix column indexes used in the polynomial of the candidate model
     * 
     * @return Vector of the column indexes
     */
    const VectorU16& combination() const { return _combination; }

    /**
     * @brief Get the vector of the calculated coefficients corresponding to the polynomial variables of the candidate model
     * 
     * @return Vector of the coefficients
     */
    const VectorXd& bestCoeffs() const { return _bestCoeffs; }

    /**
     * @brief Get the value of the external criterion evaluation for the candidate model
     * 
     * @return External criterion value
     */
    double evaluation() const { return _evaluation; }

    /**
     * @brief Set the %combination vector of the X matrix column indexes used in the polynomial of the candidate model by rvalue reference
     * 
     * @param combination Vector of the column indexes
     */
    void setCombination(VectorU16&& combination) { _combination = std::move(combination); }

    /**
     * @brief Set the vector of the X matrix column indexes used in the polynomial of the candidate model by lvalue reference
     * 
     * @param combination Vector of the column indexes
     */
    void setCombination(const VectorU16& combination) { _combination = combination; }

    /**
     * @brief Set the vector of the calculated coefficients corresponding to the polynomial variables of the candidate model
     * 
     * @param bestCoeffs Vector of the coefficients
     */
    void setBestCoeffs(VectorXd&& bestCoeffs) { _bestCoeffs = std::move(bestCoeffs);}

    /**
     * @brief Set the value of the external criterion evaluation for the candidate model
     * 
     * @param evaluation External criterion value
     */
    void setEvaluation(double evaluation) { _evaluation = evaluation; }

    /**
     * @brief Overloaded comparison operator < for the two candidate models
     * 
     * @param comb Combination object of the second candidate model
     * @return True if the left candidate model has a lower external criterion value than the right candidate model, otherwise false
     */
    bool operator<(const Combination& comb) const { return _evaluation < comb._evaluation; }

    /**
     * @brief Conver json object with trained model info to the Combination object
     *
     * @param v Json value which stores info about trained model
     * @return Combination initialized from json structure
     */
    friend GMDH_API Combination tag_invoke(boost::json::value_to_tag<Combination>, boost::json::value const& v) {
        VectorU16 bestColsIndexes;
        std::vector<double> coeffs;
        auto& o = v.as_object();
        auto& combination = o.at("combination").as_array();
        auto& bestCoeffs = o.at("bestCoeffs").as_array();
        
        for (auto i : combination)
            bestColsIndexes.push_back(i.as_int64());
        for (auto i : bestCoeffs)
            coeffs.push_back(i.as_double());
        return {
            std::move(bestColsIndexes),
            Map<VectorXd>(coeffs.data(), coeffs.size())
        };
    }

    /**
     * @brief Convert trained model info to the json value
     *
     * @param v Reference to the json value variable to which the conversion result will be written
     * @param comb Combination type variable with info about the trained model
     */
    friend GMDH_API void tag_invoke(boost::json::value_from_tag, boost::json::value& v, Combination const& comb)
    {
        v = boost::json::object{
            {"combination", comb.combination()},
            {"bestCoeffs", comb.bestCoeffs()},
        };
    }
};

/// @brief A type definition for storing the set of trained models
using VectorC = std::vector<Combination>;

/// @brief A type definition for iterator to the storage of the set of trained models
using IterC = VectorC::iterator;

/// @brief A type definition for const iterator to the storage of the set of trained models
using cIterC = VectorC::const_iterator;
};