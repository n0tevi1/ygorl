// pybind11 entry point for ygorl._core.
//
// Every Duel method releases the GIL before taking the duel's mutex. Python
// card/script sources re-acquire it inside the callback. This ordering avoids
// the deadlock where one thread holds the GIL waiting on the duel mutex while
// another holds the mutex waiting on the GIL.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

#include "core_backend.h"
#include "duel_pool.h"

namespace py = pybind11;
using namespace ygorl;

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

    py::class_<Duel>(m, "Duel", "One ygopro-core duel handle (OCG_* API).")
        .def(py::init([](std::array<uint64_t, 4> seed, uint64_t flags, py::tuple team1,
                         py::tuple team2, py::object cards, py::object scripts) {
                 return std::make_unique<Duel>(seed, flags, to_player(team1), to_player(team2),
                                               to_card_source(cards), to_script_source(scripts));
             }),
             py::arg("seed"), py::arg("flags"), py::arg("team1"), py::arg("team2"),
             py::arg("cards"), py::arg("scripts"))
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
        .def_property_readonly("closed", &Duel::closed);

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

    m.attr("DUEL_STATUS_END") = static_cast<int>(OCG_DUEL_STATUS_END);
    m.attr("DUEL_STATUS_AWAITING") = static_cast<int>(OCG_DUEL_STATUS_AWAITING);
    m.attr("DUEL_STATUS_CONTINUE") = static_cast<int>(OCG_DUEL_STATUS_CONTINUE);
    m.attr("LOG_TYPE_ERROR") = static_cast<int>(OCG_LOG_TYPE_ERROR);
    m.attr("LOG_TYPE_FROM_SCRIPT") = static_cast<int>(OCG_LOG_TYPE_FROM_SCRIPT);
    m.attr("LOG_TYPE_FOR_DEBUG") = static_cast<int>(OCG_LOG_TYPE_FOR_DEBUG);
    m.attr("LOG_TYPE_UNDEFINED") = static_cast<int>(OCG_LOG_TYPE_UNDEFINED);
}
