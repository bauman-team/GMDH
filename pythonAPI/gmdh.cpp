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
    
    py::class_<GMDH::splitted_data>(m, "splitted_data")
        .def_readwrite("x_train", &GMDH::splitted_data::x_train)
        .def_readwrite("x_test", &GMDH::splitted_data::x_test)
        .def_readwrite("y_train", &GMDH::splitted_data::y_train)
        .def_readwrite("y_test", &GMDH::splitted_data::y_test); 


    py::class_<GMDH::Criterion>(m, "Criterion");


    py::class_<GMDH::RegularityCriterionTS, GMDH::Criterion>(m, "RegularityCriterionTS")
        .def(py::init<double>())
        .def("calculate", &GMDH::RegularityCriterionTS::calculate);


    py::class_<GMDH::RegularityCriterion, GMDH::RegularityCriterionTS, GMDH::Criterion>(m, "RegularityCriterion")
        .def(py::init<double, bool, int>())
        .def("calculate", &GMDH::RegularityCriterion::calculate);


    py::class_<GMDH::GMDH>(m, "GMDH");


    py::class_<GMDH::COMBI, GMDH::GMDH>(m, "COMBI")
        .def(py::init<>())
        .def("save", &GMDH::COMBI::save)
        .def("load", &GMDH::COMBI::load)
        .def("predict", static_cast<double (GMDH::COMBI::*) (const Eigen::RowVectorXd&) const>(&GMDH::COMBI::predict))
        .def("predict", static_cast<Eigen::VectorXd (GMDH::COMBI::*) (const Eigen::MatrixXd&) const>(&GMDH::COMBI::predict))
        .def("fit", &GMDH::COMBI::fit)
        .def("getBestPolymon", &GMDH::COMBI::getBestPolymon);


    //m.def("polynomailFeatures", &polynomailFeatures);
    m.def("convertToTimeSeries", &GMDH::convertToTimeSeries);
    m.def("splitTsData", &GMDH::splitTsData);
    m.def("splitData", &GMDH::splitData);

}