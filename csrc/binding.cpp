// pybind11 entry point for ygorl._core.
//
// Every Duel method releases the GIL before taking the duel's mutex. Python
// card/script sources re-acquire it inside the callback. This ordering avoids
// the deadlock where one thread holds the GIL waiting on the duel mutex while
// another holds the mutex waiting on the GIL.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <unordered_map>

#include "core_backend.h"
#include "duel_pool.h"
#include "host.h"
#include "host_pool.h"
#include "privileged.h"
#include "event_encoder.h"

namespace py = pybind11;
using namespace ygorl;

namespace ygorl {
void bind_event_history(py::module_& m);  // event_binding.cpp
}

namespace {

// Card source backed by a Python callable: code -> None | tuple of 12 fields
// (code, alias, setcodes, type, level, attribute, race, attack, defense, lscale, rscale, link_marker).
class PyCardSource final : public CardSource {
public:
    explicit PyCardSource(py::function fn) : fn_(std::move(fn)) {}
    ~PyCardSource() override {
        py::gil_scoped_acquire gil;
        fn_ = py::function();
    }
    bool read(uint32_t code, OCG_CardData* out) override {
        py::gil_scoped_acquire gil;
        py::object result = fn_(code);
        if (result.is_none()) return false;
        auto t = result.cast<py::tuple>();
        if (t.size() != 12) throw py::value_error("card source must return a 12-tuple or None");
        auto& setcodes = setcodes_[code];
        setcodes.clear();
        for (auto s : t[2].cast<std::vector<uint16_t>>())
            if (s != 0) setcodes.push_back(s);
        setcodes.push_back(0);
        *out = OCG_CardData{};
        out->code = t[0].cast<uint32_t>();
        out->alias = t[1].cast<uint32_t>();
        out->setcodes = setcodes.data();
        out->type = t[3].cast<uint32_t>();
        out->level = t[4].cast<uint32_t>();
        out->attribute = t[5].cast<uint32_t>();
        out->race = t[6].cast<uint64_t>();
        out->attack = t[7].cast<int32_t>();
        out->defense = t[8].cast<int32_t>();
        out->lscale = t[9].cast<uint32_t>();
        out->rscale = t[10].cast<uint32_t>();
        out->link_marker = t[11].cast<uint32_t>();
        return true;
    }

private:
    py::function fn_;
    std::unordered_map<uint32_t, std::vector<uint16_t>> setcodes_;  // stable buffers per code
};

// Script source backed by a Python callable: name -> None | bytes.
class PyScriptSource final : public ScriptSource {
public:
    explicit PyScriptSource(py::function fn) : fn_(std::move(fn)) {}
    ~PyScriptSource() override {
        py::gil_scoped_acquire gil;
        fn_ = py::function();
    }
    bool load(OCG_Duel duel, const char* name) override {
        std::string content;
        {
            py::gil_scoped_acquire gil;
            py::object result = fn_(std::string(name));
            if (result.is_none()) return false;
            content = result.cast<std::string>();
        }
        arena::Resume in_core;  // the duel's arena (if any) while the core compiles the script
        return OCG_LoadScript(duel, content.data(), static_cast<uint32_t>(content.size()), name) != 0;
    }

private:
    py::function fn_;
};

std::shared_ptr<CardSource> to_card_source(py::object obj) {
    if (py::isinstance<CardDatabase>(obj)) return obj.cast<std::shared_ptr<CardDatabase>>();
    if (py::isinstance<py::function>(obj)) return std::make_shared<PyCardSource>(obj.cast<py::function>());
    throw py::type_error("cards must be a CardDatabase or a callable(code) -> tuple | None");
}

std::shared_ptr<ScriptSource> to_script_source(py::object obj) {
    if (py::isinstance<ScriptDirectory>(obj)) return obj.cast<std::shared_ptr<ScriptDirectory>>();
    if (py::isinstance<py::function>(obj)) return std::make_shared<PyScriptSource>(obj.cast<py::function>());
    throw py::type_error("scripts must be a ScriptDirectory or a callable(name) -> bytes | None");
}

PlayerOptions to_player(const py::tuple& t) {
    if (t.size() != 3) throw py::value_error("player options must be (starting_lp, starting_draw, draw_per_turn)");
    return {t[0].cast<uint32_t>(), t[1].cast<uint32_t>(), t[2].cast<uint32_t>()};
}

template <typename F>
py::bytes bytes_without_gil(F&& f) {
    std::string s;
    {
        py::gil_scoped_release release;
        s = f();
    }
    return py::bytes(s);
}

py::tuple action_tuple(const host::Action& a) {
    const auto& l = a.card.loc;
    return py::make_tuple(static_cast<int>(a.kind), a.index, a.has_card, a.card.code,
                          py::make_tuple(l.controller, l.location, l.sequence, l.position), a.description, a.value);
}

py::list action_list(const std::vector<host::Action>& acts) {
    py::list out;
    for (const auto& a : acts) out.append(action_tuple(a));
    return out;
}

py::array_t<int32_t> to_array(const std::vector<int32_t>& v, std::vector<py::ssize_t> shape) {
    py::array_t<int32_t> arr(shape);
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int32_t));
    return arr;
}

py::dict observation_dict(const host::Observation& o) {
    py::dict d;
    d["cards"] = to_array(o.cards, {host::N_CARDS, host::F_CARD});
    d["globals"] = to_array(o.globals, {host::G_GLOBAL});
    d["actions"] = to_array(o.actions, {host::MAX_OPTIONS, host::A_ACTION});
    d["action_mask"] = to_array(o.action_mask, {host::MAX_OPTIONS});
    if (o.has_events) {
        const auto n = static_cast<py::ssize_t>(o.event_mask.size());
        d["events"] = to_array(o.events, {n, host::E_EVENT});
        d["event_mask"] = to_array(o.event_mask, {n});
    }
    return d;
}

// Training-only opponent ground truth; kept out of observation_dict on purpose (the actor never sees it).
py::dict privileged_dict(const host::Privileged& p) {
    py::dict d;
    d["op_hand"] = to_array(p.op_hand, {host::P_HAND, host::P_COLS});
    d["op_deck"] = to_array(p.op_deck, {host::P_DECK, host::P_COLS});
    d["op_extra"] = to_array(p.op_extra, {host::P_EXTRA, host::P_COLS});
    d["op_set"] = to_array(p.op_set, {host::P_SET, host::P_COLS});
    d["op_removed"] = to_array(p.op_removed, {host::P_REMOVED, host::P_COLS});
    d["counts"] = to_array(p.counts, {host::P_COUNTS});
    return d;
}

// Owns a DecisionState plus the record it was decoded from (for fuzz tests against actions.py).
py::dict result_dict(const host::PoolEvent& e) {
    py::dict d;
    d["winner"] = e.winner < 0 ? py::object(py::none()) : py::object(py::int_(e.winner));
    d["reason"] = e.reason;
    d["win_reason"] = e.win_reason < 0 ? py::object(py::none()) : py::object(py::int_(e.win_reason));
    d["turns"] = e.turns;
    d["lp"] = py::make_tuple(e.lp[0], e.lp[1]);
    d["decisions"] = e.decisions;
    py::list responses;
    for (const auto& r : e.responses) responses.append(py::bytes(r));
    d["responses"] = responses;
    d["error"] = e.error;
    return d;
}

class PyDecisionState {
public:
    PyDecisionState(py::bytes record, std::shared_ptr<CardDatabase> cards) : cards_(std::move(cards)) {
        std::string r = record;
        auto d = host::decode_decision(reinterpret_cast<const uint8_t*>(r.data()), r.size());
        if (!d) throw py::value_error("not a decodable decision record");
        state_ = std::make_unique<host::DecisionState>(std::move(*d), cards_.get());
    }
    py::list actions() { return action_list(state_->actions()); }
    py::object step(size_t index) {
        if (state_->step(index)) return py::bytes(state_->response());
        return py::none();
    }
    bool done() const { return state_->done(); }

private:
    std::shared_ptr<CardDatabase> cards_;
    std::unique_ptr<host::DecisionState> state_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "ygorl native core: bindings for edo9300/ygopro-core (C++17 + pybind11)";
    m.def("hello", []() { return "ygorl._core"; }, "Smoke-test function.");
    m.def(
        "ocg_version",
        []() {
            int major = 0, minor = 0;
            OCG_GetVersion(&major, &minor);
            return py::make_tuple(major, minor);
        },
        "Return (major, minor) of the linked ygopro-core (OCG_GetVersion).");

    py::class_<CardDatabase, std::shared_ptr<CardDatabase>>(m, "CardDatabase",
        "Immutable-after-fill in-memory card table used by the core's card reader.")
        .def(py::init<>())
        .def("add", &CardDatabase::add, py::arg("code"), py::arg("alias"), py::arg("setcodes"),
             py::arg("type"), py::arg("level"), py::arg("attribute"), py::arg("race"),
             py::arg("attack"), py::arg("defense"), py::arg("lscale"), py::arg("rscale"),
             py::arg("link_marker"))
        .def("__len__", &CardDatabase::size)
        .def("__contains__", [](const CardDatabase& db, uint32_t code) { return db.find(code) != nullptr; })
        .def("get", [](const CardDatabase& db, uint32_t code) -> py::object {
            const CardRecord* r = db.find(code);
            if (!r) return py::none();
            std::vector<uint16_t> sc(r->setcodes.begin(), r->setcodes.end() - 1);
            const auto& d = r->data;
            return py::make_tuple(d.code, d.alias, sc, d.type, d.level, d.attribute, d.race,
                                  d.attack, d.defense, d.lscale, d.rscale, d.link_marker);
        }, "Return the stored 12-tuple for `code`, or None.");

    py::class_<ScriptDirectory, std::shared_ptr<ScriptDirectory>>(m, "ScriptDirectory",
        "Loads card scripts by file name from an ordered list of directories (first match wins).")
        .def(py::init<std::vector<std::string>>(), py::arg("directories"))
        .def("find", &ScriptDirectory::find, py::arg("name"))
        .def("read", [](const ScriptDirectory& s, const std::string& name) -> py::object {
            auto c = s.read(name);
            if (!c) return py::none();
            return py::bytes(*c);
        }, py::arg("name"))
        .def_property_readonly("directories", &ScriptDirectory::directories)
        .def("__len__", &ScriptDirectory::size);

    py::class_<DuelSnapshot, std::shared_ptr<DuelSnapshot>>(
        m, "DuelSnapshot", "Opaque core state of one duel (Duel.snapshot); restorable only into that duel.")
        .def_property_readonly("nbytes", &DuelSnapshot::nbytes);

    m.attr("ARENA_AVAILABLE") = arena::available();

    py::class_<Duel>(m, "Duel", "One ygopro-core duel handle (OCG_* API).")
        .def(py::init([](std::array<uint64_t, 4> seed, uint64_t flags, py::tuple team1,
                         py::tuple team2, py::object cards, py::object scripts, bool snapshots) {
                 auto card_source = to_card_source(cards);
                 auto script_source = to_script_source(scripts);
                 PlayerOptions p1 = to_player(team1), p2 = to_player(team2);
                 py::gil_scoped_release release;
                 return std::make_unique<Duel>(seed, flags, p1, p2, std::move(card_source),
                                               std::move(script_source), snapshots);
             }),
             py::arg("seed"), py::arg("flags"), py::arg("team1"), py::arg("team2"),
             py::arg("cards"), py::arg("scripts"), py::arg("snapshots") = false,
             "snapshots=True keeps the core's state in a private arena so snapshot()/restore() work (T2.8).")
        .def("snapshot", [](Duel& d) {
            std::shared_ptr<const DuelSnapshot> snap;
            {
                py::gil_scoped_release release;
                snap = d.snapshot();
            }
            return std::const_pointer_cast<DuelSnapshot>(snap);
        }, "Copy the duel's complete core state.")
        .def("restore", [](Duel& d, const DuelSnapshot& snap) {
            py::gil_scoped_release release;
            d.restore(snap);
        }, py::arg("snapshot"), "Return to a snapshot taken from this duel (ValueError for another duel's).")
        .def_property_readonly("snapshots_enabled", [](const Duel& d) {
            py::gil_scoped_release release;  // takes the duel's mutex (GIL before mutex would deadlock)
            return d.snapshots_enabled();
        })
        .def("arena_escapes", &Duel::arena_escapes, py::call_guard<py::gil_scoped_release>(),
             "Allocations that escaped the arena while the core ran (must be 0 for exact snapshots).")
        .def("arena_bytes", &Duel::arena_bytes, py::call_guard<py::gil_scoped_release>(), "Bytes of the duel's arena in use.")
        .def("load_script", &Duel::load_script, py::arg("name"),
             py::call_guard<py::gil_scoped_release>())
        .def("new_card", &Duel::new_card, py::arg("team"), py::arg("duelist"), py::arg("code"),
             py::arg("controller"), py::arg("location"), py::arg("sequence"), py::arg("position"),
             py::call_guard<py::gil_scoped_release>())
        .def("start", &Duel::start, py::call_guard<py::gil_scoped_release>())
        .def("process", &Duel::process, py::call_guard<py::gil_scoped_release>(),
             "Run the core until it needs a response or ends; returns OCG_DUEL_STATUS_*. Releases the GIL.")
        .def("get_message", [](Duel& d) { return bytes_without_gil([&] { return d.get_message(); }); })
        .def("set_response", [](Duel& d, py::bytes response) {
            std::string r = response;
            py::gil_scoped_release release;
            d.set_response(r);
        }, py::arg("response"))
        .def("query_count", &Duel::query_count, py::arg("team"), py::arg("location"),
             py::call_guard<py::gil_scoped_release>())
        .def("query", [](Duel& d, uint32_t flags, uint8_t con, uint32_t loc, uint32_t seq, uint32_t oseq) {
            return bytes_without_gil([&] { return d.query(flags, con, loc, seq, oseq); });
        }, py::arg("flags"), py::arg("controller"), py::arg("location"), py::arg("sequence"),
           py::arg("overlay_sequence") = 0)
        .def("query_location", [](Duel& d, uint32_t flags, uint8_t con, uint32_t loc) {
            return bytes_without_gil([&] { return d.query_location(flags, con, loc); });
        }, py::arg("flags"), py::arg("controller"), py::arg("location"))
        .def("query_field", [](Duel& d) { return bytes_without_gil([&] { return d.query_field(); }); })
        .def("pop_logs", [](Duel& d) {
            std::vector<LogEntry> logs;
            {
                py::gil_scoped_release release;
                logs = d.pop_logs();
            }
            py::list out;
            for (auto& e : logs) out.append(py::make_tuple(e.type, py::bytes(e.text)));
            return out;
        }, "Return and clear [(OCG_LOG_TYPE_*, bytes)] emitted by the core.")
        .def("close", &Duel::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("closed", [](const Duel& d) {
            py::gil_scoped_release release;
            return d.closed();
        });

    py::class_<DuelPool>(m, "DuelPool",
        "Advances many duels on a pool of worker threads (env i runs on thread i % num_threads).")
        .def(py::init([](size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
                         std::shared_ptr<ScriptDirectory> scripts) {
                 return std::make_unique<DuelPool>(num_envs, num_threads, cards, scripts);
             }),
             py::arg("num_envs"), py::arg("num_threads"), py::arg("cards"), py::arg("scripts"))
        .def("start", [](DuelPool& pool, int env, std::array<uint64_t, 4> seed, uint64_t flags, py::tuple team1,
                         py::tuple team2, DeckLists decks) {
            auto t1 = to_player(team1), t2 = to_player(team2);
            py::gil_scoped_release release;
            pool.start(env, seed, flags, t1, t2, std::move(decks));
        }, py::arg("env"), py::arg("seed"), py::arg("flags"), py::arg("team1"), py::arg("team2"), py::arg("decks"),
           "Asynchronously create env's duel from [(main, extra), (main, extra)] and run it to its first stop.")
        .def("respond", [](DuelPool& pool, int env, py::bytes response) {
            std::string r = response;
            py::gil_scoped_release release;
            pool.respond(env, std::move(r));
        }, py::arg("env"), py::arg("response"), "Asynchronously answer env's pending decision and run to the next stop.")
        .def("recv", [](DuelPool& pool, size_t min_results, int timeout_ms) {
            std::vector<PoolResult> results;
            {
                py::gil_scoped_release release;
                results = pool.recv(min_results, timeout_ms);
            }
            py::list out;
            for (auto& r : results) {
                py::list logs;
                for (auto& e : r.logs) logs.append(py::make_tuple(e.type, py::bytes(e.text)));
                out.append(py::make_tuple(r.env_id, r.status, py::bytes(r.buffer), logs, r.error));
            }
            return out;
        }, py::arg("min_results") = 1, py::arg("timeout_ms") = -1,
           "Return finished jobs as [(env, status, buffer, logs, error)], waiting for at least min_results.")
        .def("close_env", &DuelPool::close_env, py::arg("env"), py::call_guard<py::gil_scoped_release>())
        .def("pending", &DuelPool::pending)
        .def_property_readonly("num_envs", &DuelPool::num_envs)
        .def_property_readonly("num_threads", &DuelPool::num_threads);

    py::class_<PyDecisionState>(m, "DecisionState", "C++ mirror of ygorl.engine.actions (for cross-checking).")
        .def(py::init<py::bytes, std::shared_ptr<CardDatabase>>(), py::arg("record"), py::arg("cards"))
        .def("actions", &PyDecisionState::actions)
        .def("step", &PyDecisionState::step, py::arg("index"))
        .def_property_readonly("done", &PyDecisionState::done);

    py::class_<host::HostDuel>(m, "HostDuel", "A duel driven entirely in C++: tracker, actions and encoder.")
        .def(py::init([](std::shared_ptr<CardDatabase> cards, std::shared_ptr<ScriptDirectory> scripts,
                         const std::vector<uint32_t>& vocab, size_t event_length) {
                 auto h = std::make_unique<host::HostDuel>(cards, scripts, std::make_shared<host::Vocab>(vocab));
                 h->set_event_length(event_length);
                 return h;
             }),
             py::arg("cards"), py::arg("scripts"), py::arg("vocab"), py::arg("event_length") = 0)
        .def("start", [](host::HostDuel& h, std::array<uint64_t, 4> seed, uint64_t flags, py::tuple t1, py::tuple t2,
                         DeckLists decks, uint32_t max_turns, uint32_t max_decisions) {
            auto p1 = to_player(t1), p2 = to_player(t2);
            py::gil_scoped_release release;
            h.start(seed, flags, p1, p2, decks, max_turns, max_decisions);
        })
        .def("player", &host::HostDuel::player)
        .def("done", &host::HostDuel::done)
        .def("actions", [](host::HostDuel& h) { return action_list(h.actions()); })
        .def("act", [](host::HostDuel& h, size_t index) {
            py::gil_scoped_release release;
            h.act(index);
        }, py::arg("index"))
        .def("observe", [](host::HostDuel& h) {
            host::Observation o;
            {
                py::gil_scoped_release release;
                h.observe(o);
            }
            return observation_dict(o);
        })
        .def("observe_privileged", [](host::HostDuel& h) {
            host::Privileged p;
            {
                py::gil_scoped_release release;
                h.observe_privileged(p);
            }
            return privileged_dict(p);
        }, "Training-only opponent ground truth (docs/encoding.md); never feed it to the actor.")
        .def("result", [](host::HostDuel& h) {
            const auto& t = h.tracker();
            py::dict d;
            int w = t.winner();
            d["winner"] = w < 0 ? py::object(py::none()) : py::object(py::int_(w));
            d["reason"] = t.reason();
            d["win_reason"] = t.win_reason() < 0 ? py::object(py::none()) : py::object(py::int_(t.win_reason()));
            d["turns"] = t.turn();
            d["lp"] = py::make_tuple(t.lp()[0], t.lp()[1]);
            d["decisions"] = t.decisions();
            py::list responses;
            for (const auto& r : t.responses()) responses.append(py::bytes(r));
            d["responses"] = responses;
            d["retries"] = t.retries();
            d["error"] = t.error();
            return d;
        });

    py::class_<host::HostPool>(m, "HostPool",
        "Vectorized env with the step loop, action states and encoder in C++ (env i on thread i % threads).")
        .def(py::init([](size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
                         std::shared_ptr<ScriptDirectory> scripts, const std::vector<uint32_t>& vocab, bool privileged,
                         size_t event_length) {
                 return std::make_unique<host::HostPool>(num_envs, num_threads, cards, scripts,
                                                         std::make_shared<host::Vocab>(vocab), privileged,
                                                         event_length);
             }),
             py::arg("num_envs"), py::arg("num_threads"), py::arg("cards"), py::arg("scripts"), py::arg("vocab"),
             py::arg("privileged") = false, py::arg("event_length") = 0)
        .def("reset", [](host::HostPool& p, int env, std::array<uint64_t, 4> seed, uint64_t flags, py::tuple t1,
                         py::tuple t2, DeckLists decks, uint32_t max_turns, uint32_t max_decisions) {
            host::PoolJob job;
            job.seed = seed;
            job.flags = flags;
            job.team1 = to_player(t1);
            job.team2 = to_player(t2);
            job.decks = std::move(decks);
            job.max_turns = max_turns;
            job.max_decisions = max_decisions;
            py::gil_scoped_release release;
            p.reset(env, std::move(job));
        })
        .def("step", &host::HostPool::step, py::arg("env"), py::arg("action"), py::call_guard<py::gil_scoped_release>())
        .def("recv", [](host::HostPool& p, size_t min_results, int timeout_ms) {
            std::vector<host::PoolEvent> events;
            {
                py::gil_scoped_release release;
                events = p.recv(min_results, timeout_ms);
            }
            py::list out;
            for (const auto& e : events) {
                // 6th element: training-mode ground truth, None in inference mode and at game end
                py::object priv = e.has_privileged && !e.done ? py::object(privileged_dict(e.privileged)) : py::none();
                if (e.done)
                    out.append(py::make_tuple(e.env_id, true, e.player, py::none(), result_dict(e), priv));
                else
                    out.append(py::make_tuple(e.env_id, false, e.player, observation_dict(e.obs), py::none(), priv));
            }
            return out;
        }, py::arg("min_results") = 1, py::arg("timeout_ms") = -1)
        .def("pending", &host::HostPool::pending);

    bind_event_history(m);

    m.attr("DUEL_STATUS_END") = static_cast<int>(OCG_DUEL_STATUS_END);
    m.attr("DUEL_STATUS_AWAITING") = static_cast<int>(OCG_DUEL_STATUS_AWAITING);
    m.attr("DUEL_STATUS_CONTINUE") = static_cast<int>(OCG_DUEL_STATUS_CONTINUE);
    m.attr("LOG_TYPE_ERROR") = static_cast<int>(OCG_LOG_TYPE_ERROR);
    m.attr("LOG_TYPE_FROM_SCRIPT") = static_cast<int>(OCG_LOG_TYPE_FROM_SCRIPT);
    m.attr("LOG_TYPE_FOR_DEBUG") = static_cast<int>(OCG_LOG_TYPE_FOR_DEBUG);
    m.attr("LOG_TYPE_UNDEFINED") = static_cast<int>(OCG_LOG_TYPE_UNDEFINED);
}
