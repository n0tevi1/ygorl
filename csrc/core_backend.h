// Thin C++ wrapper over the edo9300/ygopro-core C API (OCG_*).
//
// This layer is deliberately Python-free so the M2 thread pool can reuse it.
// Card data and scripts are provided through two small interfaces:
//   CardSource   -> answers OCG_DataReader callbacks
//   ScriptSource -> answers OCG_ScriptReader callbacks (must call OCG_LoadScript)
// Native implementations (CardDatabase, ScriptDirectory) are immutable after
// construction/population and therefore safe to share across threads.
//
// Swapping the rule core means rewriting this file, core_backend.cpp and the
// message decoder; see docs/engine.md.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arena.h"
#include "ocgapi.h"

namespace ygorl {

// Engine steps (OCG_DuelProcess calls) allowed between two decisions before a duel is stopped as an engine loop:
// a core that keeps processing without ever asking a player would otherwise hang its worker thread (the decision
// limit never fires). Real chains take a few thousand steps at most. Mirror of engine.duel.MAX_ENGINE_STEPS;
// settable for tests (_core.set_max_engine_steps).
inline std::atomic<uint32_t> g_max_engine_steps{100000};

class CardSource {
public:
    virtual ~CardSource() = default;
    // Fill `out` for `code`; return false if unknown (the core then sees a blank card).
    // `out->setcodes` must stay valid until done() is called for the same pointer.
    virtual bool read(uint32_t code, OCG_CardData* out) = 0;
    virtual void done(OCG_CardData* /*data*/) {}
};

class ScriptSource {
public:
    virtual ~ScriptSource() = default;
    // Read script `name` (e.g. "c89631139.lua", "constant.lua") and hand it to
    // OCG_LoadScript. Return false if the script does not exist or fails to load.
    virtual bool load(OCG_Duel duel, const char* name) = 0;
};

struct CardRecord {
    OCG_CardData data{};
    std::vector<uint16_t> setcodes;  // zero-terminated, as the core expects
};

// In-memory card table, filled once (e.g. from cards.cdb via Python) and then read-only.
class CardDatabase final : public CardSource {
public:
    void add(uint32_t code, uint32_t alias, const std::vector<uint16_t>& setcodes, uint32_t type,
             uint32_t level, uint32_t attribute, uint64_t race, int32_t attack, int32_t defense,
             uint32_t lscale, uint32_t rscale, uint32_t link_marker);
    const CardRecord* find(uint32_t code) const;
    size_t size() const { return cards_.size(); }
    const std::unordered_map<uint32_t, CardRecord>& all() const { return cards_; }
    bool read(uint32_t code, OCG_CardData* out) override;

private:
    std::unordered_map<uint32_t, CardRecord> cards_;
};

// Looks scripts up by file name in an ordered list of directories (first match wins).
class ScriptDirectory final : public ScriptSource {
public:
    explicit ScriptDirectory(std::vector<std::string> directories);
    std::optional<std::string> find(const std::string& name) const;
    std::optional<std::string> read(const std::string& name) const;
    bool load(OCG_Duel duel, const char* name) override;
    const std::vector<std::string>& directories() const { return directories_; }
    size_t size() const { return index_.size(); }

private:
    std::vector<std::string> directories_;
    std::unordered_map<std::string, std::string> index_;  // file name -> full path
};

struct PlayerOptions {
    uint32_t starting_lp = 8000;
    uint32_t starting_draw = 5;
    uint32_t draw_per_turn = 1;
};

struct LogEntry {
    int type;
    std::string text;
};

// Complete core state of a duel at one point (T2.8); restorable only into the duel that took it.
struct DuelSnapshot {
    std::unique_ptr<arena::Image> image;
    std::vector<LogEntry> logs;
    size_t nbytes() const { return image ? image->size() : 0; }
};

// One duel = one core handle. Every public method is serialized by a mutex, so a
// Duel may be driven from any thread (but never concurrently by two).
class Duel {
public:
    Duel(const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
         const PlayerOptions& team2, std::shared_ptr<CardSource> cards,
         std::shared_ptr<ScriptSource> scripts, bool snapshots = false);
    ~Duel();
    Duel(const Duel&) = delete;
    Duel& operator=(const Duel&) = delete;

    bool load_script(const std::string& name);
    void new_card(uint8_t team, uint8_t duelist, uint32_t code, uint8_t controller, uint32_t location,
                  uint32_t sequence, uint32_t position);
    void start();
    int process();  // OCG_DUEL_STATUS_*
    std::string get_message();
    void set_response(const std::string& response);
    uint32_t query_count(uint8_t team, uint32_t location);
    std::string query(uint32_t flags, uint8_t controller, uint32_t location, uint32_t sequence,
                      uint32_t overlay_sequence);
    std::string query_location(uint32_t flags, uint8_t controller, uint32_t location);
    std::string query_field();
    std::vector<LogEntry> pop_logs();
    void close();
    bool closed() const;

    // Snapshots (duels created with snapshots=true): the core's whole state lives in a
    // private arena (csrc/arena.h), so taking and restoring a snapshot is a memory copy.
    bool snapshots_enabled() const;
    std::shared_ptr<const DuelSnapshot> snapshot();
    void restore(const DuelSnapshot& snapshot);  // std::invalid_argument if taken from another duel
    uint64_t arena_escapes() const;  // take the duel's mutex: close() may run on another thread
    size_t arena_bytes() const;

    // Callback error captured at the C boundary; rethrown by the caller-facing method.
    void capture_error(std::exception_ptr e);
    void rethrow_pending();

private:
    void ensure_open() const;  // also rejects re-entrant calls from this duel's own callbacks

    // RAII around a call into the core: activates the duel's arena and marks the duel busy, so a
    // callback that calls back into the same duel (close, process, ...) gets an error instead of
    // freeing memory the core is still using.
    class CoreCall {
    public:
        explicit CoreCall(Duel& d) : duel_(d), scope_(d.arena_.get()) { duel_.in_core_ = true; }
        ~CoreCall() { duel_.in_core_ = false; }
        CoreCall(const CoreCall&) = delete;
        CoreCall& operator=(const CoreCall&) = delete;

    private:
        Duel& duel_;
        arena::Scope scope_;
    };
    static void card_reader(void* payload, uint32_t code, OCG_CardData* data);
    static void card_reader_done(void* payload, OCG_CardData* data);
    static int script_reader(void* payload, OCG_Duel duel, const char* name);
    static void log_handler(void* payload, const char* text, int type);

    std::unique_ptr<arena::Arena> arena_;  // null unless snapshots are enabled
    OCG_Duel handle_ = nullptr;
    std::shared_ptr<CardSource> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::vector<LogEntry> logs_;
    std::exception_ptr pending_;
    mutable std::recursive_mutex mutex_;
    bool in_core_ = false;
};

}  // namespace ygorl
