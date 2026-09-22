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
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ocgapi.h"

namespace ygorl {

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

// One duel = one core handle. Every public method is serialized by a mutex, so a
// Duel may be driven from any thread (but never concurrently by two).
class Duel {
public:
    Duel(const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
         const PlayerOptions& team2, std::shared_ptr<CardSource> cards,
         std::shared_ptr<ScriptSource> scripts);
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
    bool closed() const { return handle_ == nullptr; }

    // Callback error captured at the C boundary; rethrown by the caller-facing method.
    void capture_error(std::exception_ptr e);
    void rethrow_pending();

private:
    void ensure_open() const;
    static void card_reader(void* payload, uint32_t code, OCG_CardData* data);
    static void card_reader_done(void* payload, OCG_CardData* data);
    static int script_reader(void* payload, OCG_Duel duel, const char* name);
    static void log_handler(void* payload, const char* text, int type);

    OCG_Duel handle_ = nullptr;
    std::shared_ptr<CardSource> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::vector<LogEntry> logs_;
    std::exception_ptr pending_;
    std::recursive_mutex mutex_;
};

}  // namespace ygorl
