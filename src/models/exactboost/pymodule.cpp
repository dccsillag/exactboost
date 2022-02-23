#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

#include "exactboost.hpp"


#ifdef PROFILE
#  warning "Compiling with profiling!"
#endif


PYBIND11_MODULE(model, m) {
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<BaseMetric> basemetric(m, "BaseMetric");
    py::class_<AUC>(m, "AUC", basemetric).def(py::init<>())
        .def(py::pickle([](const AUC& p) { return py::make_tuple(); },
                        [](const py::tuple& t) { return AUC(); }))
        ;
    py::class_<KS>(m, "KS", basemetric).def(py::init<>())
        .def(py::pickle([](const KS& p) { return py::make_tuple(); },
                        [](const py::tuple& t) { return KS(); }))
        ;
    py::class_<PaK>(m, "PaK", basemetric).def(py::init<length_t>())
        .def(py::pickle([](const PaK& p) { return py::make_tuple(p.k); },
                        [](const py::tuple& t) { return PaK(t[0].cast<length_t>()); }))
        ;

    py::class_<Stump>(m, "Stump")
        .def(py::init<length_t, float, float, float>())
        .def_readonly("feature", &Stump::feature)
        .def_property_readonly("xi", [](const Stump& s) { return s.base_stump.xi; })
        .def_property_readonly("a", [](const Stump& s) { return s.base_stump.a; })
        .def_property_readonly("b", [](const Stump& s) { return s.base_stump.b; })
        .def("__repr__", [](const Stump& s) { std::stringstream ss; ss << s; return ss.str(); })
        .def(py::pickle(
            [](const Stump& s) {
                return py::make_tuple(s.feature, s.base_stump.xi, s.base_stump.a, s.base_stump.b);
            },
            [](const py::tuple& t) {
                return Stump(t[0].cast<length_t>(),
                             t[1].cast<float>(), t[2].cast<float>(), t[3].cast<float>());
            }))
        ;

    py::class_<ExactBoost>(m, "ExactBoost")
        .def(py::init<BaseMetric&, float, unsigned int, unsigned int, float, float>(),
             "metric"_a, "margin"_a=0.05F,
             "n_estimators"_a=250, "n_rounds"_a=50,
             "observation_subsampling"_a=0.2, "feature_subsampling"_a=0.2)
        .def("fit",
             static_cast<void (ExactBoost::*)(const xt::xtensor<float, 2>&,
                                              const xt::xtensor<uint8_t, 1>&,
                                              bool)>
             (&ExactBoost::fit),
             "X"_a, "y"_a, "interaction"_a=true)
        .def("predict",
             static_cast<xt::xtensor<float, 1> (ExactBoost::*)(const xt::xtensor<float, 2>&)>
             (&ExactBoost::predict))
        .def_property_readonly("stumps", &ExactBoost::get_stumps)
        .def_property_readonly("margin", [](const ExactBoost& s) { return s.margin; })
        .def_property_readonly("n_estimators", [](const ExactBoost& s) { return s.n_estimators; })
        .def_property_readonly("n_rounds", [](const ExactBoost& s) { return s.n_rounds; })
        .def_property_readonly("observation_subsampling", [](const ExactBoost& s) { return s.observation_subsampling; })
        .def_property_readonly("feature_subsampling", [](const ExactBoost& s) { return s.feature_subsampling; })
        .def(py::pickle(
            [](const ExactBoost& eb) {
                return py::make_tuple(eb.metric, eb.margin,
                                      eb.n_estimators, eb.n_rounds,
                                      eb.observation_subsampling, eb.feature_subsampling,
                                      eb.feature_scales,
                                      eb.get_stumps());
            },
            [](const py::tuple& t) {
                return ExactBoost(*t[0].cast<BaseMetric*>(), t[1].cast<float>(),
                                  t[2].cast<unsigned int>(), t[3].cast<unsigned int>(),
                                  t[4].cast<float>(), t[5].cast<float>(),
                                  t[6].cast<std::vector<float>>(),
                                  t[7].cast<std::vector<std::vector<Stump>>>());
            }))
        ;
}
