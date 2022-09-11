#include <pybind11/eigen.h>
// to convert C++ STL containers to python list
#include <pybind11/stl.h>
#include "../src/combi.h"
#include "../src/ria.h"

namespace py = pybind11;

PYBIND11_MODULE(gmdhpy, m)
{
    using namespace std;
    using namespace pybind11::literals;
    m.doc() = "Group method of data handling";  // TODO: add main documentation and for all methods
    
    py::class_<GMDH::SplittedData>(m, "splitted_data")
        .def_readwrite("x_train", &GMDH::SplittedData::xTrain)
        .def_readwrite("x_test", &GMDH::SplittedData::xTest)
        .def_readwrite("y_train", &GMDH::SplittedData::yTrain)
        .def_readwrite("y_test", &GMDH::SplittedData::yTest); 

    py::enum_<GMDH::Solver>(m, "Solver")
        .value("fast", GMDH::Solver::fast)
        .value("balanced", GMDH::Solver::balanced)
        .value("accurate", GMDH::Solver::accurate);

    py::enum_<GMDH::CriterionType>(m, "CriterionType")
        .value("regularity", GMDH::CriterionType::regularity)
        .value("sym_regularity", GMDH::CriterionType::symRegularity)
        .value("stability", GMDH::CriterionType::stability)
        .value("sym_stability", GMDH::CriterionType::symStability)
        .value("unbiasedOutputs", GMDH::CriterionType::unbiasedOutputs)
        .value("sym_unbiased_outputs", GMDH::CriterionType::symUnbiasedOutputs)
        .value("unbiased_coeffs", GMDH::CriterionType::unbiasedCoeffs)
        .value("absolute_stability", GMDH::CriterionType::absoluteStability)
        .value("sym_absolute_stability", GMDH::CriterionType::symAbsoluteStability);

    py::enum_<GMDH::PolynomialType>(m, "PolynomialType")
        .value("linear", GMDH::PolynomialType::linear)
        .value("linear_cov", GMDH::PolynomialType::linear_cov)
        .value("quadratic", GMDH::PolynomialType::quadratic);

    py::class_<GMDH::Criterion>(m, "Criterion")
        .def(py::init<>())
        .def(py::init<GMDH::CriterionType, GMDH::Solver>(), 
            "", 
            "criterionType"_a, "solver"_a = GMDH::Solver::balanced);

    py::class_<GMDH::ParallelCriterion, GMDH::Criterion>(m, "ParallelCriterion")
        .def(py::init<GMDH::CriterionType, GMDH::CriterionType, double, GMDH::Solver>(), 
            "", 
            "criterion_type"_a, "second_criterion_type"_a, "alpha"_a = 0.5, "solver"_a = GMDH::Solver::balanced);

    py::class_<GMDH::SequentialCriterion, GMDH::Criterion>(m, "SequentialCriterion")
        .def(py::init<GMDH::CriterionType, GMDH::CriterionType, GMDH::Solver>(), 
            "", 
            "criterion_type"_a, "second_criterion_type"_a, "solver"_a = GMDH::Solver::balanced);

    py::class_<GMDH::GmdhModel>(m, "GmdhModel");

    py::class_<GMDH::MULTI, GMDH::GmdhModel>(m, "Multi")
        .def(py::init<>())
        .def("save", &GMDH::MULTI::save,
            "",
            "path"_a)
        .def("load", &GMDH::MULTI::load,
            "",
            "path"_a)
        .def("predict", static_cast<double (GMDH::MULTI::*) (const Eigen::RowVectorXd&) const>(&GMDH::GmdhModel::predict),
            "",
            "x"_a)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MULTI::*) (const Eigen::MatrixXd&) const>(&GMDH::MULTI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MULTI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "k_best"_a, "test_size"_a = 0.5, "p_average"_a = 1, 
            "n_jobs"_a = 1, "verbose"_a = 0,  "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::MULTI::getBestPolynomial);

    py::class_<GMDH::COMBI, GMDH::MULTI>(m, "Combi")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save,
            "",
            "path"_a)
        .def("load", &GMDH::COMBI::load,
            "",
            "path"_a)
        .def("predict", static_cast<double (GMDH::COMBI::*) (const Eigen::RowVectorXd&) const>(&GMDH::GmdhModel::predict),
            "",
            "x"_a)
        .def("predict", static_cast<Eigen::VectorXd (GMDH::COMBI::*) (const Eigen::MatrixXd&) const>(&GMDH::COMBI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::COMBI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "test_size"_a = 0.5, "p_average"_a = 1, 
            "n_jobs"_a = 1, "verbose"_a = 0,  "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::COMBI::getBestPolynomial);

    py::class_<GMDH::MIA, GMDH::GmdhModel>(m, "Mia")
        .def(py::init<>())
        .def("save", &GMDH::MIA::save,
            "",
            "path"_a)
        .def("load", &GMDH::MIA::load,
            "",
            "path"_a)
        .def("predict", static_cast<double (GMDH::MIA::*) (const Eigen::RowVectorXd&) const>(&GMDH::GmdhModel::predict),
            "",
            "x"_a)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MIA::*) (const Eigen::MatrixXd&) const>(&GMDH::MIA::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "k_best"_a, "polynomial_type"_a = GMDH::PolynomialType::quadratic, 
            "test_size"_a = 0.5, "p_average"_a = 1, "n_jobs"_a = 1, "verbose"_a = 0, "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::MIA::getBestPolynomial);

    py::class_<GMDH::RIA, GMDH::MIA>(m, "Ria")
        .def(py::init<>())
        .def("save", &GMDH::RIA::save)
        .def("load", &GMDH::RIA::load)
        .def("predict", static_cast<double (GMDH::RIA::*) (const Eigen::RowVectorXd&) const>(&GMDH::GmdhModel::predict))
        .def("predict", static_cast<Eigen::VectorXd(GMDH::RIA::*) (const Eigen::MatrixXd&) const>(&GMDH::RIA::predict))
        .def("fit", &GMDH::RIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "k_best"_a, "polynomial_type"_a = GMDH::PolynomialType::quadratic,
            "test_size"_a = 0.5, "p_average"_a = 1, "n_jobs"_a = 1, "verbose"_a = 0, "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::RIA::getBestPolynomial);

    m.def("time_series_transformation", &GMDH::timeSeriesTransformation,
        "",
        "x"_a, "lags"_a);
    m.def("split_data", &GMDH::splitData,
        "",
        "x"_a, "y"_a, "test_size"_a = 0.2, "shuffle"_a = false, "random_state"_a = 0);
}