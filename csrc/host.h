// C++ host layer: decision decoding, action state machines, duel tracker and
// observation encoder. It mirrors the Python reference implementation
// (ygorl/engine/messages.py, actions.py, duel.py DuelTracker, env/encoding.py)
// element for element; tests/test_cpp_host.py checks the two against each other.
#pragma once

#include <array>
#include <map>
#include <tuple>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core_backend.h"

namespace ygorl::host {

// Same order as ygorl.env.encoding.ACTION_KINDS.
enum Kind : uint8_t {
    SUMMON, SPSUMMON, REPOSITION, MSET, SSET, ACTIVATE, BATTLE_PHASE, END_PHASE, SHUFFLE,
    ATTACK, MAIN2, YES, NO, OPTION, NUMBER, RPS, CHAIN, PASS, POSITION, SELECT,
    UNSELECT, FINISH, CANCEL, COUNTER, SORT, DEFAULT, PLACE, RACE, ATTRIBUTE, DECLARE,
};

struct Loc {
    uint8_t controller = 0;
    uint8_t location = 0;
    uint32_t sequence = 0;
    uint32_t position = 0;
};

struct CardRef {
    uint32_t code = 0;
    Loc loc;
};

struct Action {
    Kind kind = SUMMON;
    int32_t index = -1;
    bool has_card = false;
    CardRef card;
    uint64_t description = 0;
    int64_t value = 0;
};

struct ChainOption {
    CardRef card;
    uint64_t description = 0;
};

// One decoded decision message (only the fields its type uses are filled).
struct Decision {
    uint8_t type = 0;
    uint8_t player = 0;
    // IDLECMD / BATTLECMD
    std::vector<CardRef> summonable, spsummonable, repositionable, msetable, ssetable;
    std::vector<ChainOption> activatable;
    std::vector<std::pair<CardRef, bool>> attackable;
    bool can_battle_phase = false, can_end_phase = false, can_shuffle = false, can_main2 = false;
    // EFFECTYN / YESNO
    bool has_card = false;
    CardRef card;
    uint64_t description = 0;
    // OPTION / ANNOUNCE_NUMBER / ANNOUNCE_CARD (opcodes)
    std::vector<uint64_t> options;
    // SELECT_CARD / TRIBUTE / SUM / UNSELECT / SORT / COUNTER / CHAIN
    bool cancelable = false, finishable = false, forced = false;
    uint32_t min = 0, max = 0;
    std::vector<CardRef> cards, unselectable;
    std::vector<uint32_t> params;  // tribute release_param / sum param / counter counts
    std::vector<ChainOption> chains;
    bool exact = false;
    uint32_t target = 0;
    std::vector<uint32_t> must_params;
    uint16_t counter_type = 0, count = 0;
    // PLACE / DISFIELD
    uint32_t flag = 0;
    uint8_t place_count = 0;
    // POSITION
    uint32_t position_code = 0;
    uint8_t positions = 0;
    // ANNOUNCE_RACE / ATTRIB
    uint64_t available = 0;
    uint8_t announce_count = 0;
};

bool is_decision_type(uint8_t type);
// Decode one record ([type][payload]); nullopt if the type is not a decision or the payload is malformed.
std::optional<Decision> decode_decision(const uint8_t* data, size_t size);
// Mirror of messages.is_hidden_from / hide_private: zero the codes of cards the decider cannot see
// in SELECT_CARD / SELECT_TRIBUTE / SELECT_UNSELECT_CARD (as EDOPro's server does).
bool is_hidden_from(uint8_t viewer, const Loc& loc);
void hide_private(Decision& d);

// Step-wise builder of the response to one decision (mirror of actions.DecisionState).
class DecisionState {
public:
    DecisionState(Decision decision, const CardDatabase* cards);
    const Decision& decision() const { return d_; }
    const std::vector<Action>& actions();
    // Apply action `index`; returns true once the response is complete (see response()).
    bool step(size_t index);
    bool done() const { return done_; }
    const std::string& response() const { return response_; }
    size_t picked_count() const { return picked_.size(); }

private:
    std::vector<Action> legal() const;
    bool apply(const Action& a);  // returns true when complete
    // helpers
    bool tribute_can_reach(const std::vector<int>& chosen) const;
    bool sum_feasible(const std::vector<int>& chosen) const;
    bool sum_complete(const std::vector<int>& chosen) const;
    bool sum_exact_feasible(const std::vector<int>& chosen) const;
    bool sum_greater_valid(const std::vector<int>& chosen) const;
    bool sum_greater_dead(const std::vector<int>& chosen) const;
    std::vector<uint32_t> place_zones() const;
    void set_cards_response();

    Decision d_;
    const CardDatabase* cards_;
    std::vector<int64_t> picked_;
    std::vector<Action> actions_;
    bool actions_valid_ = false;
    bool done_ = false;
    std::string response_;
};

bool is_declarable(const CardRecord& card, const std::vector<uint64_t>& opcodes);

struct TrackerConfig {
    uint32_t starting_lp = 8000;
    uint32_t max_turns = 200;
    uint32_t max_decisions = 20000;
};

// Mirror of engine.duel.DuelTracker, in engine-player order.
class Tracker {
public:
    Tracker(TrackerConfig cfg, const CardDatabase* cards);
    void on_buffer(const std::string& buf, int status);
    bool done() const { return done_; }
    bool awaiting() const { return !done_ && state_ != nullptr; }
    // Current actions (checks the decision limit first); empty if the duel had to stop.
    const std::vector<Action>* actions();
    // Apply an action; returns the response when the decision is complete, else nullptr.
    const std::string* act(size_t index);
    void stop(const std::string& reason, const std::string& error = "");

    const Decision* decision() const { return state_ ? &state_->decision() : nullptr; }
    const DecisionState* state() const { return state_.get(); }
    uint32_t turn() const { return turn_; }
    uint8_t turn_player() const { return turn_player_; }
    uint16_t phase() const { return phase_; }
    const std::array<int64_t, 2>& lp() const { return lp_; }
    uint32_t decisions() const { return decisions_; }
    const std::vector<std::string>& responses() const { return responses_; }
    const std::string& reason() const { return reason_; }
    const std::string& error() const { return error_; }
    int winner() const;  // engine player, -1 for draw / none
    int win_reason() const { return win_reason_; }
    uint32_t retries() const { return retries_; }
    uint32_t unknown_messages() const { return unknown_; }
    // Indices of the current actions that only undo the previous step (docs/encoding.md 「撤销类空操作」;
    // mirror of DuelTracker._undo); never all of them.
    std::vector<size_t> undo() const;

private:
    TrackerConfig cfg_;
    const CardDatabase* cards_;
    std::array<int64_t, 2> lp_;
    uint32_t turn_ = 0;
    uint8_t turn_player_ = 0;
    uint16_t phase_ = 0;
    bool done_ = false;
    std::string reason_, error_;
    int engine_winner_ = -1;
    int win_reason_ = -1;
    uint32_t retries_ = 0, unknown_ = 0, consecutive_retries_ = 0, decisions_ = 0;
    std::optional<Decision> last_decision_;
    std::unique_ptr<DecisionState> state_;
    std::vector<std::string> responses_;
    // no-op undo tracking: the player inside a command just started from a menu (-1: none), and the last
    // select / unselect of a SELECT_UNSELECT_CARD; any game event clears both
    int inside_ = -1;
    struct Toggle {
        int player = -1;
        Kind kind = SELECT;
        uint32_t code = 0;
        Loc loc;
    };
    std::optional<Toggle> toggle_;
    // repeated activation limit (rule 3, mirror of DuelTracker._activations): menu activations of one effect
    // this turn, keyed by (player, has card, code, controller, location, sequence, description)
    using ActKey = std::tuple<int, bool, uint32_t, uint8_t, uint8_t, uint32_t, uint64_t>;
    static ActKey act_key(int player, const Action& a);
    std::map<ActKey, uint32_t> activations_;
    uint32_t activations_turn_ = UINT32_MAX;
};

// An effect activated this many times from a menu in one turn is masked there (docs/encoding.md 「撤销类空操作」
// rule 3; mirror of duel.MAX_MENU_ACTIVATIONS).
constexpr uint32_t MAX_MENU_ACTIVATIONS = 8;

// Observation encoder (mirror of env.encoding.ObservationEncoder, docs/encoding.md).
constexpr int N_CARDS = 160, F_CARD = 23, G_GLOBAL = 22, MAX_OPTIONS = 128, A_ACTION = 10;

class Vocab {
public:
    explicit Vocab(const std::vector<uint32_t>& passwords);
    int32_t index(uint32_t password) const;
    static constexpr int32_t UNKNOWN = 1, FIRST_INDEX = 2;

private:
    std::unordered_map<uint32_t, int32_t> index_;
};

struct Observation {
    std::vector<int32_t> cards, globals, actions, action_mask;
    bool has_events = false;  // event token stream (T2.4, event_encoder.h): [L, E_EVENT] + [L]
    std::vector<int32_t> events, event_mask;
};

class EventHistory;  // event_encoder.h

void encode(Duel& core, const Tracker& tracker, const std::vector<Action>& actions, const CardDatabase& cards,
            const Vocab& vocab, Observation& out);

struct Privileged;  // privileged.h (T2.5)

// A single duel driven entirely in C++ (used by tests and by the pool).
class HostDuel {
public:
    HostDuel(std::shared_ptr<CardDatabase> cards, std::shared_ptr<ScriptSource> scripts,
             std::shared_ptr<const Vocab> vocab);
    void start(const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
               const PlayerOptions& team2,
               const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>& decks,
               uint32_t max_turns, uint32_t max_decisions);
    bool started() const { return tracker_ != nullptr; }  // start() completed
    bool done() const { return tracker_ && tracker_->done(); }
    const std::vector<Action>& actions();
    int player() const;
    void act(size_t index);
    void observe(Observation& out);
    // Training-only opponent ground truth (privileged.cpp); never part of observe().
    void observe_privileged(Privileged& out);
    const Tracker& tracker() const {
        require_started();
        return *tracker_;
    }
    // Keep the last n event tokens per viewer from the next start() on (0 = no event stream).
    void set_event_length(size_t n) { event_length_ = n; }

private:
    void advance();
    void require_started() const;

    std::shared_ptr<CardDatabase> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::shared_ptr<const Vocab> vocab_;
    std::unique_ptr<Duel> core_;
    std::unique_ptr<Tracker> tracker_;
    std::vector<Action> empty_;
    size_t event_length_ = 0;
    std::shared_ptr<EventHistory> events_;
};

}  // namespace ygorl::host
