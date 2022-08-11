#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// to convert C++ STL containers to python list
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "../src/combi.h"

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
        .value("accurate", GMDH::Solver::accurate)
        .export_values();

    py::class_<GMDH::Criterion>(m, "Criterion");

    py::class_<GMDH::RegularityCriterionTS, GMDH::Criterion>(m, "RegularityCriterionTS")
        .def(py::init<double, GMDH::Solver>())
        .def("calculate", &GMDH::RegularityCriterionTS::calculate);


    py::class_<GMDH::RegularityCriterion, GMDH::RegularityCriterionTS, GMDH::Criterion>(m, "RegularityCriterion")
        .def(py::init<double, GMDH::Solver, bool, int>())
        .def("calculate", &GMDH::RegularityCriterion::calculate);


    py::class_<GMDH::GMDH>(m, "GMDH");


    py::class_<GMDH::COMBI, GMDH::GMDH>(m, "COMBI")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save)
        .def("load", &GMDH::COMBI::load)
        .def("predict", static_cast<double (GMDH::COMBI::*) (const Eigen::RowVectorXd&) const>(&GMDH::COMBI::predict))
        .def("predict", static_cast<Eigen::VectorXd (GMDH::COMBI::*) (const Eigen::MatrixXd&) const>(&GMDH::COMBI::predict))
        .def("fit", &GMDH::COMBI::fit, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("getBestPolymon", &GMDH::COMBI::getBestPolynomial);


    //m.def("polynomailFeatures", &polynomailFeatures);
    m.def("convertToTimeSeries", &GMDH::convertToTimeSeries);
    m.def("splitTsData", &GMDH::splitTimeSeries);
    m.def("splitData", &GMDH::splitData);
}