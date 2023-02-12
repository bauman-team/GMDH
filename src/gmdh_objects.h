namespace GMDH {

using namespace Eigen;

using VectorU16 = std::vector<uint16_t>;
using IterU16 = VectorU16::iterator;
using VectorVu16 = std::vector<VectorU16>;
using PairDVXd = std::pair<double, VectorXd>;
using PairMVXd = std::pair<MatrixXd, VectorXd>;
using VectorI = std::vector<int>;

class Criterion;

struct GMDH_API SplittedData {
    MatrixXd xTrain;
    MatrixXd xTest;
    VectorXd yTrain;
    VectorXd yTest;
};

class GMDH_API Combination { 
    VectorU16 _combination;
    VectorXd _bestCoeffs;
    double _evaluation;
public:
    Combination() {}
    Combination(VectorU16 comb) : _combination(comb) {} 
    Combination(VectorU16&& comb, VectorXd&& coeffs) : _combination(std::move(comb)), _bestCoeffs(std::move(coeffs)) {} 
    const VectorU16& combination() const { return _combination; }
    const VectorXd& bestCoeffs() const { return _bestCoeffs; }
    double evaluation() const { return _evaluation; }

    void setCombination(VectorU16&& combination) { _combination = std::move(combination); }
    void setCombination(const VectorU16& combination) { _combination = combination; }
    void setBestCoeffs(VectorXd&& bestCoeffs) { _bestCoeffs = std::move(bestCoeffs);}
    void setEvaluation(double evaluation) { _evaluation = evaluation; }

    bool operator<(const Combination& comb) const { return _evaluation < comb._evaluation; }

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

    friend GMDH_API void tag_invoke(boost::json::value_from_tag, boost::json::value& v, Combination const& comb)
    {
        v = boost::json::object{
            {"combination", comb.combination()},
            {"bestCoeffs", comb.bestCoeffs()},
        };
    }

};

using VectorC = std::vector<Combination>;
using IterC = VectorC::iterator;
using cIterC = VectorC::const_iterator;
};