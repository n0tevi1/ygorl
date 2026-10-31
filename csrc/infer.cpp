// Snapshot-opponent inference in libtorch (ygorl._infer): the traced ActPath graph runs entirely in
// C++ with the GIL released, so several collection threads forward concurrently without the Python
// dispatch storm that dominated the collect loop (docs/benchmarks.md "收集循环剖析", 2026-10-30).
//
// The artifact is produced by tools/export_snapshot_actor.py (format "ygorl-actpath-trace-v1"): the
// graph takes the collated observation dict and returns masked action logits; it accepts any batch
// size and prefix-sliced widths, so this side narrows the fixed-width inputs to the true prefix
// (the semantics of nets.policy.trim_padding) before running it.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ATen/Parallel.h>
#include <torch/python.h>
#include <torch/script.h>

namespace py = pybind11;

namespace {

class OpponentRunner {
public:
    explicit OpponentRunner(const std::string &path) {
        module_ = torch::jit::load(path);
        module_.eval();
    }

    // Copy a state_dict into the loaded graph's parameters (in place; the caller passes host tensors).
    // Called between segments, never next to a concurrent forward.
    void set_state(const py::dict &state) {
        for (const auto &np : module_.named_parameters()) {
            if (!state.contains(np.name.c_str())) continue;
            at::Tensor src = state[np.name.c_str()].cast<at::Tensor>();
            np.value.copy_(src);
        }
    }

    at::Tensor forward(at::Tensor cards, at::Tensor globals, at::Tensor actions, at::Tensor action_mask,
                       at::Tensor events, at::Tensor event_mask, int64_t n_cards, int64_t n_actions,
                       int64_t n_events, at::Tensor selection_history, at::Tensor attack_history) {
        // all Python objects are converted on entry; nothing below touches the interpreter
        const bool has_sel = selection_history.defined() && selection_history.numel() > 0;
        const bool has_atk = attack_history.defined() && attack_history.numel() > 0;
        py::gil_scoped_release nogil;  // the whole point: the graph runs without the interpreter
        c10::InferenceMode guard;
        c10::impl::GenericDict obs(c10::StringType::get(), c10::TensorType::get());
        obs.insert("cards", cards.narrow(1, 0, n_cards));
        obs.insert("globals", globals);
        obs.insert("actions", actions.narrow(1, 0, n_actions));
        obs.insert("action_mask", action_mask.narrow(1, 0, n_actions));
        obs.insert("events", events.narrow(1, 0, n_events));
        obs.insert("event_mask", event_mask.narrow(1, 0, n_events));
        if (has_sel) obs.insert("selection_history", selection_history);
        if (has_atk) obs.insert("attack_history", attack_history);
        at::Tensor logits = module_.forward({obs}).toTensor();
        return at::softmax(logits, /*dim=*/-1).to(at::kFloat);
    }

private:
    torch::jit::Module module_;
};

}  // namespace

PYBIND11_MODULE(_infer, m) {
    m.doc() = "Snapshot-opponent inference on traced actor graphs (GIL-free)";
    py::class_<OpponentRunner>(m, "OpponentRunner")
        .def(py::init<const std::string &>(), py::arg("path"))
        .def("set_state", &OpponentRunner::set_state, py::arg("state"))
        .def("forward", &OpponentRunner::forward, py::arg("cards"), py::arg("globals"), py::arg("actions"),
             py::arg("action_mask"), py::arg("events"), py::arg("event_mask"), py::arg("n_cards"),
             py::arg("n_actions"), py::arg("n_events"), py::arg("selection_history") = py::none(),
             py::arg("attack_history") = py::none());
}
