// pybind11 bindings of the event token stream (T2.4): _core.EventHistory, for
// cross-checking csrc/event_encoder.cpp against ygorl.env.events.EventHistory.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "event_encoder.h"

namespace py = pybind11;

namespace ygorl {

namespace {

// Owns what the C++ history points to (card database, vocabulary).
class PyEventHistory {
public:
    PyEventHistory(std::shared_ptr<CardDatabase> cards, const std::vector<uint32_t>& vocab, size_t length,
                   int64_t starting_lp)
        : cards_(std::move(cards)),
          vocab_(std::make_shared<host::Vocab>(vocab)),
          history_(cards_.get(), vocab_.get(), length, starting_lp) {}

    void feed(py::bytes buf) {
        std::string b = buf;
        py::gil_scoped_release release;
        history_.feed(b);
    }

    py::tuple encode(int viewer) const {
        if (viewer != 0 && viewer != 1) throw py::value_error("viewer must be 0 or 1");
        std::vector<int32_t> events, mask;
        history_.encode(viewer, events, mask);
        const auto n = static_cast<py::ssize_t>(mask.size());
        py::array_t<int32_t> e({n, static_cast<py::ssize_t>(host::E_EVENT)});
        py::array_t<int32_t> m({n});
        if (!events.empty()) std::memcpy(e.mutable_data(), events.data(), events.size() * sizeof(int32_t));
        if (!mask.empty()) std::memcpy(m.mutable_data(), mask.data(), mask.size() * sizeof(int32_t));
        return py::make_tuple(e, m);
    }

    int64_t hand_count(int player) const { return history_.hand_count(player); }
    int field_count(int player) const { return history_.field_count(player); }
    size_t length() const { return history_.length(); }

private:
    std::shared_ptr<CardDatabase> cards_;
    std::shared_ptr<host::Vocab> vocab_;
    host::EventHistory history_;
};

}  // namespace

void bind_event_history(py::module_& m) {
    py::class_<PyEventHistory>(m, "EventHistory", "C++ mirror of ygorl.env.events.EventHistory (for cross-checking).")
        .def(py::init<std::shared_ptr<CardDatabase>, const std::vector<uint32_t>&, size_t, int64_t>(),
             py::arg("cards"), py::arg("vocab"), py::arg("length") = host::DEFAULT_EVENT_LENGTH,
             py::arg("starting_lp") = 8000)
        .def("feed", &PyEventHistory::feed, py::arg("buffer"), "Consume one engine message buffer.")
        .def("encode", &PyEventHistory::encode, py::arg("viewer"), "Return (events [L, E], event_mask [L]).")
        .def("hand_count", &PyEventHistory::hand_count, py::arg("player"))
        .def("field_count", &PyEventHistory::field_count, py::arg("player"))
        .def_property_readonly("length", &PyEventHistory::length);
}

}  // namespace ygorl
