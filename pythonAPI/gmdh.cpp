#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// to convert C++ STL containers to python list
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "../src/combi.h"
#include "../src/mia.h"

namespace py = pybind11;

PYBIND11_MODULE(gmdhpy, m)
{
    using namespace std;
    
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
        .def(py::init<GMDH::CriterionType, GMDH::Solver>())
        .def("getClassName", &GMDH::Criterion::getClassName)
        .def("calculate", &GMDH::Criterion::calculate);

    py::class_<GMDH::ParallelCriterion, GMDH::Criterion>(m, "ParallelCriterion")
        .def(py::init<GMDH::CriterionType, GMDH::CriterionType, double, GMDH::Solver>())
        .def("getClassName", &GMDH::ParallelCriterion::getClassName)
        .def("calculate", &GMDH::ParallelCriterion::calculate);

    py::class_<GMDH::SequentialCriterion, GMDH::Criterion>(m, "SequentialCriterion")
        .def(py::init<GMDH::CriterionType, GMDH::CriterionType, GMDH::Solver>())
        .def("getClassName", &GMDH::SequentialCriterion::getClassName)
        .def("calculate", &GMDH::SequentialCriterion::calculate)
        .def("recalculate", &GMDH::SequentialCriterion::recalculate);

    py::class_<GMDH::GMDH>(m, "GMDH");

    py::class_<GMDH::MULTI, GMDH::GMDH>(m, "MULTI")
        .def(py::init<>())
        .def("save", &GMDH::MULTI::save)
        .def("load", &GMDH::MULTI::load)
        .def("predict", static_cast<double (GMDH::MULTI::*) (const Eigen::RowVectorXd&) const>(&GMDH::MULTI::predict))
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MULTI::*) (const Eigen::MatrixXd&) const>(&GMDH::MULTI::predict))
        .def("fit", &GMDH::MULTI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("getBestPolymon", &GMDH::MULTI::getBestPolynomial);

    py::class_<GMDH::COMBI, GMDH::MULTI>(m, "COMBI")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save)
        .def("load", &GMDH::COMBI::load)
        .def("predict", static_cast<double (GMDH::COMBI::*) (const Eigen::RowVectorXd&) const>(&GMDH::COMBI::predict))
        .def("predict", static_cast<Eigen::VectorXd (GMDH::COMBI::*) (const Eigen::MatrixXd&) const>(&GMDH::COMBI::predict))
        .def("fit", &GMDH::COMBI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("getBestPolymon", &GMDH::COMBI::getBestPolynomial);

    py::class_<GMDH::MIA, GMDH::GMDH>(m, "MIA")
        .def(py::init<>())
        .def("save", &GMDH::MIA::save)
        .def("load", &GMDH::MIA::load)
        .def("predict", static_cast<double (GMDH::MIA::*) (const Eigen::RowVectorXd&) const>(&GMDH::MIA::predict))
        .def("predict", static_cast<Eigen::VectorXd(GMDH::MIA::*) (const Eigen::MatrixXd&) const>(&GMDH::MIA::predict))
        .def("fit", &GMDH::MIA::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("getBestPolymon", &GMDH::MIA::getBestPolynomial);

    //m.def("polynomailFeatures", &polynomailFeatures);
    m.def("convertToTimeSeries", &GMDH::convertToTimeSeries);
    m.def("splitData", &GMDH::splitData);
}