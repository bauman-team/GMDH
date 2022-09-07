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
        .value("symRegularity", GMDH::CriterionType::symRegularity)
        .value("stability", GMDH::CriterionType::stability)
        .value("symStability", GMDH::CriterionType::symStability)
        .value("unbiasedOutputs", GMDH::CriterionType::unbiasedOutputs)
        .value("symUnbiasedOutputs", GMDH::CriterionType::symUnbiasedOutputs)
        .value("unbiasedCoeffs", GMDH::CriterionType::unbiasedCoeffs)
        .value("absoluteStability", GMDH::CriterionType::absoluteStability)
        .value("symAbsoluteStability", GMDH::CriterionType::symAbsoluteStability);

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
            "criterionType"_a, "secondCriterionType"_a, "alpha"_a = 0.5, "solver"_a = GMDH::Solver::balanced);

    py::class_<GMDH::SequentialCriterion, GMDH::Criterion>(m, "SequentialCriterion")
        .def(py::init<GMDH::CriterionType, GMDH::CriterionType, GMDH::Solver>(), 
            "", 
            "criterionType"_a, "secondCriterionType"_a, "solver"_a = GMDH::Solver::balanced);

    py::class_<GMDH::GmdhModel>(m, "GmdhModel");

    py::class_<GMDH::MULTI, GMDH::GmdhModel>(m, "MULTI")
        .def(py::init<>())
        .def("save", &GMDH::MULTI::save,
            "",
            "path"_a)
        .def("load", &GMDH::MULTI::load,
            "",
            "path"_a)
        /*.def("predict", static_cast<double (GMDH::MULTI::*) (const Eigen::RowVectorXd&) const>(&GMDH::MULTI::predict),
            "",
            "x"_a)*/
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MULTI::*) (const Eigen::MatrixXd&) const>(&GMDH::MULTI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MULTI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "kBest"_a, "testSize"_a = 0.5, "shuffle"_a = false, "randomSeed"_a = 0, 
            "pAverage"_a = 1, "threads"_a = 1, "verbose"_a = 0,  "limit"_a = 0)
        .def("getBestPolynomial", &GMDH::MULTI::getBestPolynomial);

    py::class_<GMDH::COMBI, GMDH::MULTI>(m, "COMBI")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save,
            "",
            "path"_a)
        .def("load", &GMDH::COMBI::load,
            "",
            "path"_a)
        /*.def("predict", static_cast<double (GMDH::COMBI::*) (const Eigen::RowVectorXd&) const>(&GMDH::COMBI::predict),
            "",
            "x"_a)*/
        .def("predict", static_cast<Eigen::VectorXd (GMDH::COMBI::*) (const Eigen::MatrixXd&) const>(&GMDH::COMBI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::COMBI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "testSize"_a = 0.5, "shuffle"_a = false, "randomSeed"_a = 0, 
            "pAverage"_a = 1, "threads"_a = 1, "verbose"_a = 0,  "limit"_a = 0)
        .def("getBestPolynomial", &GMDH::COMBI::getBestPolynomial);

    py::class_<GMDH::MIA, GMDH::GmdhModel>(m, "MIA")
        .def(py::init<>())
        .def("save", &GMDH::MIA::save,
            "",
            "path"_a)
        .def("load", &GMDH::MIA::load,
            "",
            "path"_a)
        /*.def("predict", static_cast<double (GMDH::MIA::*) (const Eigen::RowVectorXd&) const>(&GMDH::MIA::predict),
            "",
            "x"_a)*/
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MIA::*) (const Eigen::MatrixXd&) const>(&GMDH::MIA::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "kBest"_a, "polynomialType"_a = GMDH::PolynomialType::quadratic, "testSize"_a = 0.5, 
            "shuffle"_a = false, "randomSeed"_a = 0, "pAverage"_a = 1, "threads"_a = 1, "verbose"_a = 0, "limit"_a = 0)
        .def("getBestPolynomial", &GMDH::MIA::getBestPolynomial);

    py::class_<GMDH::RIA, GMDH::MIA>(m, "RIA")
        .def(py::init<>())
        .def("save", &GMDH::RIA::save)
        .def("load", &GMDH::RIA::load)
        /*.def("predict", static_cast<double (GMDH::RIA::*) (const Eigen::RowVectorXd&) const>(&GMDH::MIA::predict))*/
        .def("predict", static_cast<Eigen::VectorXd(GMDH::RIA::*) (const Eigen::MatrixXd&) const>(&GMDH::RIA::predict))
        .def("fit", &GMDH::RIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a, "kBest"_a, "polynomialType"_a = GMDH::PolynomialType::quadratic,
            "testSize"_a = 0.5, "shuffle"_a = false, "randomSeed"_a = 0, "pAverage"_a = 1, "threads"_a = 1,
            "verbose"_a = 0, "limit"_a = 0)
        .def("getBestPolynomial", &GMDH::RIA::getBestPolynomial);

    m.def("timeSeriesTransformation", &GMDH::timeSeriesTransformation,
        "",
        "x"_a, "lags"_a);
    m.def("splitData", &GMDH::splitData,
        "",
        "x"_a, "y"_a, "testSize"_a = 0.2, "shuffle"_a = false, "randomSeed"_a = 0);
}