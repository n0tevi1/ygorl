// Event token stream with response-window / abstain tokens (T2.4): C++ mirror of
// ygorl/env/events.py (spec: docs/encoding.md, 事件 token 流). tests/test_events.py
// checks the two element by element.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "host.h"

namespace ygorl::host {

constexpr int E_EVENT = 20;
constexpr size_t DEFAULT_EVENT_LENGTH = 128;

// Internal records of EventHistory.
struct EventCard {  // a card reference in a token: code plus whether each viewer may see it
    bool present = false;
    uint32_t code = 0;
    bool seen[2] = {true, true};
};
struct EventLink {
    int player = 0;
    uint32_t code = 0;
    Loc loc;
    int32_t triggers = 0;
};
struct EventWindow {
    int abstainer = 0;
    int32_t trigger = 0;
    EventCard card;
    Loc loc;
    bool responded = false;
};

class EventHistory {
public:
    using Card = EventCard;
    using Link = EventLink;
    using Window = EventWindow;

    EventHistory(const CardDatabase* cards, const Vocab* vocab, size_t length, int64_t starting_lp);
    // Consume one engine message buffer ([u32 length][u8 type][payload]...).
    void feed(const std::string& buf);
    // Last `length` tokens of `viewer` as [length, E_EVENT] row-major plus mask.
    void encode(int viewer, std::vector<int32_t>& events, std::vector<int32_t>& mask) const;
    size_t length() const { return length_; }
    int64_t hand_count(int player) const { return hand_[player & 1]; }
    int field_count(int player) const;

private:
    using Row = std::array<int32_t, E_EVENT>;

    void on_record(const uint8_t* rec, size_t len);
    void emit(int kind, int player = -1, const Card& card = {}, const Card& card2 = {}, const Loc* from = nullptr,
              const Loc* to = nullptr, int64_t v1 = 0, int64_t v2 = 0, int64_t v3 = 0, int only = -1);
    int32_t card_col(const Card& c, int viewer) const;
    Card field_card(const Loc& loc) const;
    void place(uint32_t code, const Loc& loc);
    void hand_delta(const Loc& loc, int delta);
    void close_windows();
    void abstain(int abstainer, int32_t trigger, const Card& card, const Loc& loc);

    const CardDatabase* cards_;
    const Vocab* vocab_;
    size_t length_;
    std::array<std::deque<Row>, 2> tokens_;
    uint32_t turn_ = 0;
    int turn_player_ = 0;
    uint16_t phase_ = 0;
    std::array<int64_t, 2> lp_;
    std::array<int64_t, 2> hand_{0, 0};
    // field_[con][0 = monster zone (7 slots), 1 = spell/trap zone (8 slots)][seq] = (code, position)
    struct Slot {
        bool used = false;
        uint32_t code = 0, position = 0;
    };
    std::array<std::array<std::array<Slot, 8>, 2>, 2> field_{};
    int summons_ = 0;
    std::map<int64_t, Link> links_;
    int64_t solving_ = 0;
    std::vector<Window> windows_;
};

}  // namespace ygorl::host
