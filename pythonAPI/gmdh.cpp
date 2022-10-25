#include <pybind11/eigen.h>
// to convert C++ STL containers to python list
#include <pybind11/stl.h>
#include <exception>
#include "../src/combi.h"
#include "../src/multi.h"
#include "../src/ria.h"

namespace py = pybind11;

PYBIND11_MODULE(_gmdh_core, m)
{
    using namespace std;
    using namespace pybind11::literals;
    m.doc() = "Group method of data handling";  // TODO: add main documentation and for all methods

    py::register_exception<GMDH::GmdhException>(m, "GmdhException");
    py::register_local_exception<GMDH::GmdhException>(m, "GmdhException");
    
    py::class_<GMDH::SplittedData>(m, "splitted_data")
        .def_readwrite("x_train", &GMDH::SplittedData::xTrain)
        .def_readwrite("x_test", &GMDH::SplittedData::xTest)
        .def_readwrite("y_train", &GMDH::SplittedData::yTrain)
        .def_readwrite("y_test", &GMDH::SplittedData::yTest); 

    py::enum_<GMDH::Solver>(m, "Solver")
        .value("FAST", GMDH::Solver::fast)
        .value("ACCURATE", GMDH::Solver::accurate)
        .value("BALANCED", GMDH::Solver::balanced);

    py::enum_<GMDH::CriterionType>(m, "CriterionType")
        .value("REGULARITY", GMDH::CriterionType::regularity)
        .value("SYM_REGULARITY", GMDH::CriterionType::symRegularity)
        .value("STABILITY", GMDH::CriterionType::stability)
        .value("SYM_STABILITY", GMDH::CriterionType::symStability)
        .value("UNBIASED_OUTPUTS", GMDH::CriterionType::unbiasedOutputs)
        .value("SYM_UNBIASED_OUTPUTS", GMDH::CriterionType::symUnbiasedOutputs)
        .value("UNBIASED_COEFFS", GMDH::CriterionType::unbiasedCoeffs)
        .value("ABSOLUTE_STABILITY", GMDH::CriterionType::absoluteStability)
        .value("SYM_ABSOLUTE_STABILITY", GMDH::CriterionType::symAbsoluteStability);

    py::enum_<GMDH::PolynomialType>(m, "PolynomialType")
        .value("LINEAR", GMDH::PolynomialType::linear)
        .value("LINEAR_COV", GMDH::PolynomialType::linear_cov)
        .value("QUADRATIC", GMDH::PolynomialType::quadratic);

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
    py::class_<GMDH::LinearModel, GMDH::GmdhModel>(m, "LinearModel");

    py::class_<GMDH::MULTI, GMDH::LinearModel>(m, "Multi")
        .def(py::init<>())
        .def("save", &GMDH::MULTI::save,
            "",
            "path"_a)
        .def("load", &GMDH::MULTI::load,
            "",
            "path"_a)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MULTI::*) (const Eigen::RowVectorXd&, int) const>
                        (&GMDH::GmdhModel::predict),
            "",
            "x"_a, "lags"_a = 1)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MULTI::*) (const Eigen::MatrixXd&) const>
                        (&GMDH::MULTI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MULTI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a = GMDH::Criterion(GMDH::CriterionType::regularity), 
            "k_best"_a = 3, "test_size"_a = 0.5, "p_average"_a = 1, 
            "n_jobs"_a = 1, "verbose"_a = 0,  "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::MULTI::getBestPolynomial);

    py::class_<GMDH::COMBI, GMDH::LinearModel>(m, "Combi")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save,
            "",
            "path"_a)
        .def("load", &GMDH::COMBI::load,
            "",
            "path"_a)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::COMBI::*) (const Eigen::RowVectorXd&, int) const>
                        (&GMDH::GmdhModel::predict),
            "",
            "x"_a, "lags"_a = 1)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::COMBI::*) (const Eigen::MatrixXd&) const>
                        (&GMDH::COMBI::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::COMBI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a = GMDH::Criterion(GMDH::CriterionType::regularity), 
            "test_size"_a = 0.5, "p_average"_a = 1,
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
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MIA::*) (const Eigen::RowVectorXd&, int) const>
                        (&GMDH::GmdhModel::predict),
            "",
            "x"_a, "lags"_a = 1)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MIA::*) (const Eigen::MatrixXd&) const>
                        (&GMDH::MIA::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::MIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(), 
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a = GMDH::Criterion(GMDH::CriterionType::regularity), 
            "k_best"_a = 3, "polynomial_type"_a = GMDH::PolynomialType::quadratic,
            "test_size"_a = 0.5, "p_average"_a = 1,
            "n_jobs"_a = 1, "verbose"_a = 0, "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::MIA::getBestPolynomial);

    py::class_<GMDH::RIA, GMDH::MIA>(m, "Ria")
        .def(py::init<>())
        .def("save", &GMDH::RIA::save,
            "",
            "path"_a)
        .def("load", &GMDH::RIA::load,
            "",
            "path"_a)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::RIA::*) (const Eigen::RowVectorXd&, int) const>
                        (&GMDH::GmdhModel::predict),
            "",
            "x"_a, "lags"_a = 1)
        .def("predict", static_cast<Eigen::VectorXd(GMDH::RIA::*) (const Eigen::MatrixXd&) const>
                        (&GMDH::RIA::predict),
            "",
            "x"_a)
        .def("fit", &GMDH::RIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>(),
            "This method used for training model",
            "x"_a, "y"_a, "criterion"_a = GMDH::Criterion(GMDH::CriterionType::regularity), 
            "k_best"_a = 3, "polynomial_type"_a = GMDH::PolynomialType::quadratic,
            "test_size"_a = 0.5, "p_average"_a = 1, 
            "n_jobs"_a = 1, "verbose"_a = 0, "limit"_a = 0)
        .def("get_best_polynomial", &GMDH::RIA::getBestPolynomial);

    m.def("time_series_transformation", &GMDH::timeSeriesTransformation);
    m.def("split_data", &GMDH::splitData);
}