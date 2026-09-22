#include "core_backend.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace ygorl {

// ---------------------------------------------------------------- CardDatabase

void CardDatabase::add(uint32_t code, uint32_t alias, const std::vector<uint16_t>& setcodes,
                       uint32_t type, uint32_t level, uint32_t attribute, uint64_t race,
                       int32_t attack, int32_t defense, uint32_t lscale, uint32_t rscale,
                       uint32_t link_marker) {
    CardRecord rec;
    rec.setcodes.reserve(setcodes.size() + 1);
    for (auto s : setcodes)
        if (s != 0) rec.setcodes.push_back(s);
    rec.setcodes.push_back(0);
    rec.data.code = code;
    rec.data.alias = alias;
    rec.data.type = type;
    rec.data.level = level;
    rec.data.attribute = attribute;
    rec.data.race = race;
    rec.data.attack = attack;
    rec.data.defense = defense;
    rec.data.lscale = lscale;
    rec.data.rscale = rscale;
    rec.data.link_marker = link_marker;
    auto& slot = cards_[code];
    slot = std::move(rec);
    slot.data.setcodes = slot.setcodes.data();  // node storage is stable
}

const CardRecord* CardDatabase::find(uint32_t code) const {
    auto it = cards_.find(code);
    return it == cards_.end() ? nullptr : &it->second;
}

bool CardDatabase::read(uint32_t code, OCG_CardData* out) {
    const CardRecord* rec = find(code);
    if (!rec) return false;
    *out = rec->data;
    out->setcodes = const_cast<uint16_t*>(rec->setcodes.data());
    return true;
}

// ------------------------------------------------------------- ScriptDirectory

ScriptDirectory::ScriptDirectory(std::vector<std::string> directories)
    : directories_(std::move(directories)) {
    for (const auto& dir : directories_) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".lua") continue;
            index_.emplace(entry.path().filename().string(), entry.path().string());  // first wins
        }
    }
}

std::optional<std::string> ScriptDirectory::find(const std::string& name) const {
    // Scripts may request "./script/c123.lua"-style paths; only the file name matters.
    auto it = index_.find(fs::path(name).filename().string());
    if (it == index_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string> ScriptDirectory::read(const std::string& name) const {
    auto path = find(name);
    if (!path) return std::nullopt;
    std::ifstream in(*path, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool ScriptDirectory::load(OCG_Duel duel, const char* name) {
    auto content = read(name);
    if (!content) return false;
    arena::Resume in_core;  // back into the duel's arena (if any) while the core compiles the script
    return OCG_LoadScript(duel, content->data(), static_cast<uint32_t>(content->size()), name) != 0;
}

// ------------------------------------------------------------------------ Duel

Duel::Duel(const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
           const PlayerOptions& team2, std::shared_ptr<CardSource> cards,
           std::shared_ptr<ScriptSource> scripts, bool snapshots)
    : cards_(std::move(cards)), scripts_(std::move(scripts)) {
    if (!cards_ || !scripts_) throw std::invalid_argument("card and script sources are required");
    if (snapshots) arena_ = arena::Arena::create();
    OCG_DuelOptions opts{};
    std::copy(seed.begin(), seed.end(), opts.seed);
    opts.flags = flags;
    opts.team1 = {team1.starting_lp, team1.starting_draw, team1.draw_per_turn};
    opts.team2 = {team2.starting_lp, team2.starting_draw, team2.draw_per_turn};
    opts.cardReader = &Duel::card_reader;
    opts.payload1 = this;
    opts.scriptReader = &Duel::script_reader;
    opts.payload2 = this;
    opts.logHandler = &Duel::log_handler;
    opts.payload3 = this;
    opts.cardReaderDone = &Duel::card_reader_done;
    opts.payload4 = this;
    opts.enableUnsafeLibraries = 0;
    int status;
    {
        arena::Scope in_core(arena_.get());
        status = OCG_CreateDuel(&handle_, &opts);
    }
    if (status != OCG_DUEL_CREATION_SUCCESS) {
        handle_ = nullptr;
        throw std::runtime_error("OCG_CreateDuel failed with status " + std::to_string(status));
    }
    if (pending_) {  // a card/script callback failed while the core was being set up
        {
            arena::Scope in_core(arena_.get());
            OCG_DestroyDuel(handle_);
        }
        handle_ = nullptr;
        rethrow_pending();
    }
}

Duel::~Duel() { close(); }

void Duel::close() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (handle_) {
        {
            arena::Scope in_core(arena_.get());
            OCG_DestroyDuel(handle_);
        }
        handle_ = nullptr;
    }
    arena_.reset();
}

void Duel::ensure_open() const {
    if (!handle_) throw std::runtime_error("duel is closed");
}

void Duel::capture_error(std::exception_ptr e) {
    if (!pending_) pending_ = std::move(e);
}

void Duel::rethrow_pending() {
    if (pending_) {
        auto e = std::move(pending_);
        pending_ = nullptr;
        std::rethrow_exception(e);
    }
}

// Callbacks run inside the core (and inside Lua, which is compiled as C++ and
// catches everything): exceptions must never cross this boundary.
void Duel::card_reader(void* payload, uint32_t code, OCG_CardData* data) {
    arena::Suspend host;  // callbacks allocate on the host heap (see arena.h)
    auto* self = static_cast<Duel*>(payload);
    try {
        if (!self->cards_->read(code, data)) {
            static uint16_t no_setcodes[1] = {0};
            *data = OCG_CardData{};
            data->code = code;
            data->setcodes = no_setcodes;
        }
    } catch (...) {
        self->capture_error(std::current_exception());
        static uint16_t no_setcodes[1] = {0};
        *data = OCG_CardData{};
        data->code = code;
        data->setcodes = no_setcodes;
    }
}

void Duel::card_reader_done(void* payload, OCG_CardData* data) {
    arena::Suspend host;
    auto* self = static_cast<Duel*>(payload);
    try {
        self->cards_->done(data);
    } catch (...) {
        self->capture_error(std::current_exception());
    }
}

int Duel::script_reader(void* payload, OCG_Duel duel, const char* name) {
    arena::Suspend host;  // ScriptSource::load resumes the arena around OCG_LoadScript
    auto* self = static_cast<Duel*>(payload);
    try {
        return self->scripts_->load(duel, name) ? 1 : 0;
    } catch (...) {
        self->capture_error(std::current_exception());
        return 0;
    }
}

void Duel::log_handler(void* payload, const char* text, int type) {
    arena::Suspend host;
    auto* self = static_cast<Duel*>(payload);
    try {
        self->logs_.push_back({type, text ? std::string(text) : std::string()});
    } catch (...) {
    }
}

bool Duel::load_script(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    bool ok;
    {
        arena::Scope in_core(arena_.get());
        ok = script_reader(this, handle_, name.c_str()) != 0;
    }
    rethrow_pending();
    return ok;
}

void Duel::new_card(uint8_t team, uint8_t duelist, uint32_t code, uint8_t controller,
                    uint32_t location, uint32_t sequence, uint32_t position) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    OCG_NewCardInfo info{team, duelist, code, controller, location, sequence, position};
    {
        arena::Scope in_core(arena_.get());
        OCG_DuelNewCard(handle_, &info);
    }
    rethrow_pending();
}

void Duel::start() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    {
        arena::Scope in_core(arena_.get());
        OCG_StartDuel(handle_);
    }
    rethrow_pending();
}

int Duel::process() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    int status;
    {
        arena::Scope in_core(arena_.get());
        status = OCG_DuelProcess(handle_);
    }
    rethrow_pending();
    return status;
}

std::string Duel::get_message() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    uint32_t length = 0;
    const char* buf;
    {
        arena::Scope in_core(arena_.get());
        buf = static_cast<const char*>(OCG_DuelGetMessage(handle_, &length));
    }
    return std::string(buf, buf + length);
}

void Duel::set_response(const std::string& response) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    arena::Scope in_core(arena_.get());
    OCG_DuelSetResponse(handle_, response.data(), static_cast<uint32_t>(response.size()));
}

uint32_t Duel::query_count(uint8_t team, uint32_t location) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    arena::Scope in_core(arena_.get());
    return OCG_DuelQueryCount(handle_, team, location);
}

std::string Duel::query(uint32_t flags, uint8_t controller, uint32_t location, uint32_t sequence,
                        uint32_t overlay_sequence) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    OCG_QueryInfo info{flags, controller, location, sequence, overlay_sequence};
    uint32_t length = 0;
    const char* buf;
    {
        arena::Scope in_core(arena_.get());
        buf = static_cast<const char*>(OCG_DuelQuery(handle_, &length, &info));
    }
    rethrow_pending();
    return std::string(buf, buf + length);
}

std::string Duel::query_location(uint32_t flags, uint8_t controller, uint32_t location) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    OCG_QueryInfo info{flags, controller, location, 0, 0};
    uint32_t length = 0;
    const char* buf;
    {
        arena::Scope in_core(arena_.get());
        buf = static_cast<const char*>(OCG_DuelQueryLocation(handle_, &length, &info));
    }
    rethrow_pending();
    return std::string(buf, buf + length);
}

std::string Duel::query_field() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    uint32_t length = 0;
    const char* buf;
    {
        arena::Scope in_core(arena_.get());
        buf = static_cast<const char*>(OCG_DuelQueryField(handle_, &length));
    }
    return std::string(buf, buf + length);
}

std::shared_ptr<const DuelSnapshot> Duel::snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    if (!arena_) throw std::runtime_error("this duel was created without snapshots (pass snapshots=True)");
    if (pending_) throw std::runtime_error("cannot snapshot a duel with a pending callback error");
    auto snap = std::make_shared<DuelSnapshot>();
    snap->image = arena_->capture();
    snap->logs = logs_;
    return snap;
}

void Duel::restore(const DuelSnapshot& snapshot) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ensure_open();
    if (!arena_) throw std::runtime_error("this duel was created without snapshots (pass snapshots=True)");
    if (!snapshot.image || snapshot.image->arena_id() != arena_->id())
        throw std::invalid_argument("the snapshot was taken from another duel");
    arena_->restore(*snapshot.image);
    logs_ = snapshot.logs;
    pending_ = nullptr;
}

std::vector<LogEntry> Duel::pop_logs() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<LogEntry> out;
    out.swap(logs_);
    return out;
}

}  // namespace ygorl
