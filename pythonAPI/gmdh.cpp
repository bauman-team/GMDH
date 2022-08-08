#include "../src/gmdh.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
// to convert C++ STL containers to python list
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;
using namespace GMDH;
PYBIND11_MODULE(gmdhpy, m)
{
    
    py::class_<splitted_data>(m, "splitted_data")
        .def_readwrite("x_train", &splitted_data::x_train)
        .def_readwrite("x_test", &splitted_data::x_test)
        .def_readwrite("y_train", &splitted_data::y_train)
        .def_readwrite("y_test", &splitted_data::y_test); 


    py::class_<Criterion>(m, "Criterion");


    py::class_<RegularityCriterionTS, Criterion>(m, "RegularityCriterionTS")
        .def(py::init<double>())
        .def("calculate", &RegularityCriterionTS::calculate);


    py::class_<RegularityCriterion, RegularityCriterionTS, Criterion>(m, "RegularityCriterion")
        .def(py::init<double, bool, int>())
        .def("calculate", &RegularityCriterion::calculate);


    py::class_<GMDH::GMDH>(m, "GMDH");


    py::class_<COMBI, GMDH::GMDH>(m, "COMBI")
        .def(py::init<>())
        .def("save", &COMBI::save)
        .def("load", &COMBI::load)
        .def("predict", static_cast<double (COMBI::*) (Eigen::RowVectorXd) const>(&COMBI::predict))
        .def("predict", static_cast<Eigen::VectorXd (COMBI::*) (Eigen::MatrixXd) const>(&COMBI::predict))
        .def("fit", &COMBI::fit)
        .def("getBestPolymon", &COMBI::getBestPolymon);


    //m.def("polynomailFeatures", &polynomailFeatures);
    m.def("convertToTimeSeries", &convertToTimeSeries);
    m.def("splitTsData", &splitTsData);
    m.def("splitData", &splitData);

}