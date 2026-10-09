// Event token stream: C++ mirror of ygorl/env/events.py (spec: docs/encoding.md, 事件 token 流).
#include "event_encoder.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "ocgapi_constants.h"

namespace ygorl::host {

namespace {

// Same order as ygorl.env.events.EVENT_TYPES (value = index + 1).
enum Ev : int {
    EV_DRAW = 1, EV_MOVE, EV_POS_CHANGE, EV_SET, EV_SWAP, EV_SUMMONING, EV_SPSUMMONING, EV_FLIPSUMMONING, EV_CHAINING,
    EV_CHAIN_SOLVING, EV_CHAIN_NEGATED, EV_CHAIN_DISABLED, EV_CHAIN_END, EV_NEW_TURN, EV_NEW_PHASE, EV_DAMAGE, EV_RECOVER,
    EV_PAY_LPCOST, EV_LP_UPDATE, EV_ATTACK, EV_BATTLE, EV_ATTACK_DISABLED, EV_EQUIP, EV_UNEQUIP, EV_CARD_TARGET,
    EV_CANCEL_TARGET, EV_BECOME_TARGET, EV_CARD_SELECTED, EV_RANDOM_SELECTED, EV_ADD_COUNTER, EV_REMOVE_COUNTER,
    EV_CONFIRM_CARDS, EV_CONFIRM_DECKTOP, EV_CONFIRM_EXTRATOP, EV_DECK_TOP, EV_SHUFFLE_DECK, EV_SHUFFLE_HAND,
    EV_SHUFFLE_EXTRA, EV_SHUFFLE_SET_CARD, EV_SWAP_GRAVE_DECK, EV_REVERSE_DECK, EV_FIELD_DISABLED, EV_TOSS_COIN,
    EV_TOSS_DICE, EV_HAND_RES, EV_ABSTAIN, EV_SELECTION_CHOICE,
};

// ygorl.env.events.TRIGGERS
constexpr int32_t kSearch = 1, kSpsummonDeck = 2, kSendDeckGrave = 4, kFifthSummon = 8, kAttack = 16, kOther = 32;

namespace col {
enum Col { TYPE, PLAYER, CARD, CARD2, FROM, TO = 8, VALUE1 = 12, VALUE2, VALUE3, TURN, PHASE, MY_TURN, MY_LP, OP_LP };
}

constexpr int64_t kClamp = 65535;

int64_t clamp(int64_t v, int64_t lo = 0, int64_t hi = kClamp) { return std::max(lo, std::min(hi, v)); }

int32_t bit_index(uint64_t v) {
    if (!v) return 0;
    int32_t i = 1;
    while (!(v & 1)) {
        v >>= 1;
        ++i;
    }
    return i;
}

bool faceup(uint32_t position) { return (position & POS_FACEUP) != 0; }

int32_t position_enum(uint32_t p) {
    switch (p) {
        case POS_FACEUP_ATTACK: return 1;
        case POS_FACEDOWN_ATTACK: return 2;
        case POS_FACEUP_DEFENSE: return 3;
        case POS_FACEDOWN_DEFENSE: return 4;
        default: break;
    }
    if (p & POS_FACEUP) return 5;
    if (p & POS_FACEDOWN) return 6;
    return 0;
}

int32_t location_enum(uint32_t l) {
    switch (l) {
        case LOCATION_DECK: return 1;
        case LOCATION_HAND: return 2;
        case LOCATION_MZONE: return 3;
        case LOCATION_SZONE: return 4;
        case LOCATION_GRAVE: return 5;
        case LOCATION_REMOVED: return 6;
        case LOCATION_EXTRA: return 7;
        default: return 0;
    }
}

bool is_public(const Loc& loc) {
    const uint32_t l = loc.location;
    if (l & (LOCATION_OVERLAY | LOCATION_GRAVE)) return true;
    if (l & (LOCATION_DECK | LOCATION_HAND)) return false;
    if (l == LOCATION_MZONE || l == LOCATION_SZONE || l == LOCATION_REMOVED || l == LOCATION_EXTRA)
        return faceup(loc.position);
    return false;
}

int rel(int player, int viewer) {
    if (player != 0 && player != 1) return 0;
    return player == viewer ? 1 : 2;
}

struct Truncated : std::runtime_error {
    Truncated() : std::runtime_error("truncated payload") {}
};

class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
    template <typename T>
    T get() {
        if (static_cast<size_t>(end_ - p_) < sizeof(T)) throw Truncated();
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    uint8_t u8() { return get<uint8_t>(); }
    uint16_t u16() { return get<uint16_t>(); }
    uint32_t u32() { return get<uint32_t>(); }
    int32_t i32() { return get<int32_t>(); }
    uint64_t u64() { return get<uint64_t>(); }
    void skip(size_t n) {
        if (static_cast<size_t>(end_ - p_) < n) throw Truncated();
        p_ += n;
    }
    Loc loc_info() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u32();
        l.position = u32();
        return l;
    }
    Loc loc32() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u32();
        return l;
    }
    Loc loc8() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u8();
        return l;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

Loc make_loc(uint8_t con, uint8_t location, uint32_t seq, uint32_t pos) {
    Loc l;
    l.controller = con;
    l.location = location;
    l.sequence = seq;
    l.position = pos;
    return l;
}

// Slot of a monster / spell-trap zone, or false.
bool slot_of(const Loc& loc, int& zone) {
    if (loc.controller > 1) return false;
    if (loc.location == LOCATION_MZONE && loc.sequence < 7) zone = 0;
    else if (loc.location == LOCATION_SZONE && loc.sequence < 8) zone = 1;
    else return false;
    return true;
}

}  // namespace

EventHistory::EventHistory(const CardDatabase* cards, const Vocab* vocab, size_t length, int64_t starting_lp,
                           bool selection_history)
    : cards_(cards), vocab_(vocab), length_(length), selection_history_(selection_history),
      lp_{starting_lp, starting_lp} {
    if (selection_history && !length)
        throw std::invalid_argument("selection history requires a nonempty event window");
}

void EventHistory::on_action(int player, uint8_t decision_type, const Action& action) {
    if (!selection_history_ || (action.kind != SELECT && action.kind != UNSELECT &&
                                action.kind != CANCEL && action.kind != FINISH)) return;
    auto& ordinal = selection_ordinals_.at(player);
    ordinal = std::min<uint32_t>(65535, ordinal + 1);
    Card card, owner;
    if (action.has_card) { card.present = true; card.code = action.card.code; }
    if (action.description >> 20) { owner.present = true; owner.code = action.description >> 20; }
    emit(EV_SELECTION_CHOICE, player, card, owner, action.has_card ? &action.card.loc : nullptr,
         nullptr, static_cast<int>(action.kind) + 1, decision_type, ordinal, player);
}

int EventHistory::field_count(int player) const {
    int n = 0;
    for (const auto& zone : field_[player & 1])
        for (const auto& s : zone) n += s.used ? 1 : 0;
    return n;
}

void EventHistory::encode(int viewer, std::vector<int32_t>& events, std::vector<int32_t>& mask) const {
    events.assign(length_ * E_EVENT, 0);
    mask.assign(length_, 0);
    const auto& toks = tokens_[viewer & 1];
    for (size_t i = 0; i < toks.size(); ++i) {
        std::memcpy(events.data() + i * E_EVENT, toks[i].data(), E_EVENT * sizeof(int32_t));
        mask[i] = 1;
    }
}

int32_t EventHistory::card_col(const Card& c, int viewer) const {
    if (!c.present) return 0;
    if (!c.seen[viewer]) return Vocab::UNKNOWN;
    return vocab_->index(c.code);
}

EventHistory::Card EventHistory::field_card(const Loc& loc) const {
    Card c;
    int zone;
    if (!slot_of(loc, zone)) return c;
    const Slot& s = field_[loc.controller][zone][loc.sequence];
    if (!s.used) return c;
    c.present = true;
    c.code = s.code;
    c.seen[0] = loc.controller == 0 || faceup(s.position);
    c.seen[1] = loc.controller == 1 || faceup(s.position);
    return c;
}

void EventHistory::place(uint32_t code, const Loc& loc) {
    int zone;
    if (!slot_of(loc, zone)) return;
    field_[loc.controller][zone][loc.sequence] = Slot{true, code, loc.position};
}

void EventHistory::hand_delta(const Loc& loc, int delta) {
    if (loc.location == LOCATION_HAND && loc.controller <= 1) hand_[loc.controller] += delta;
}

void EventHistory::emit(int kind, int player, const Card& card, const Card& card2, const Loc* from, const Loc* to,
                        int64_t v1, int64_t v2, int64_t v3, int only) {
    if (length_ == 0) return;
    for (int viewer = 0; viewer < 2; ++viewer) {
        if (only >= 0 && viewer != only) continue;
        Row r{};
        r[col::TYPE] = kind;
        r[col::PLAYER] = rel(player, viewer);
        r[col::CARD] = card_col(card, viewer);
        r[col::CARD2] = card_col(card2, viewer);
        for (auto [loc, base] : {std::pair<const Loc*, int>{from, col::FROM}, {to, col::TO}}) {
            if (!loc) continue;
            const bool overlay = (loc->location & LOCATION_OVERLAY) != 0;
            const int32_t location = overlay ? 8 : location_enum(loc->location);
            if (location == 0) continue;
            r[base] = rel(loc->controller, viewer);
            r[base + 1] = location;
            r[base + 2] = location == 1 ? 0 : static_cast<int32_t>(clamp(loc->sequence));
            r[base + 3] = overlay ? 0 : position_enum(loc->position);
        }
        r[col::VALUE1] = static_cast<int32_t>(v1);
        r[col::VALUE2] = static_cast<int32_t>(v2);
        r[col::VALUE3] = static_cast<int32_t>(v3);
        r[col::TURN] = static_cast<int32_t>(clamp(turn_, 0, 999));
        r[col::PHASE] = bit_index(phase_);
        r[col::MY_TURN] = turn_player_ == viewer ? 1 : 0;
        r[col::MY_LP] = static_cast<int32_t>(clamp(lp_[viewer]));
        r[col::OP_LP] = static_cast<int32_t>(clamp(lp_[1 - viewer]));
        auto& q = tokens_[viewer];
        q.push_back(r);
        if (q.size() > length_) q.pop_front();
    }
}

void EventHistory::abstain(int abstainer, int32_t trigger, const Card& card, const Loc& loc) {
    emit(EV_ABSTAIN, abstainer, card, {}, &loc, nullptr, trigger, field_count(abstainer), hand_[abstainer]);
}

void EventHistory::close_windows() {
    std::vector<Window> windows;
    windows.swap(windows_);
    for (const auto& w : windows)
        if (!w.responded) abstain(w.abstainer, w.trigger, w.card, w.loc);
}

void EventHistory::feed(const std::string& buf) {
    const auto* data = reinterpret_cast<const uint8_t*>(buf.data());
    size_t pos = 0;
    while (pos + 4 <= buf.size()) {
        uint32_t len;
        std::memcpy(&len, data + pos, 4);
        if (len == 0 || pos + 4 + len > buf.size()) break;  // corrupt framing: the rest is one undecodable message
        try {
            on_record(data + pos + 4, len);
        } catch (const Truncated&) {
            // undecodable payload: ignored (Python yields UndecodableMessage)
        }
        pos += 4 + len;
    }
}

void EventHistory::on_record(const uint8_t* rec, size_t len) {
    const uint8_t type = rec[0];
    Reader r(rec + 1, len - 1);
    auto card = [](uint32_t code, bool seen0 = true, bool seen1 = true) {
        Card c;
        c.present = true;
        c.code = code;
        c.seen[0] = seen0;
        c.seen[1] = seen1;
        return c;
    };
    switch (type) {
        case MSG_SELECT_IDLECMD:
        case MSG_SELECT_BATTLECMD:
            if (decode_decision(rec, len)) close_windows();
            break;
        case MSG_DRAW: {
            const uint8_t player = r.u8();
            const uint32_t n = r.u32();
            std::vector<std::pair<uint32_t, uint32_t>> drawn;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t code = r.u32();
                drawn.emplace_back(code, r.u32());
            }
            const Loc to = make_loc(player, LOCATION_HAND, 0, 0);
            for (const auto& [code, pos] : drawn)
                emit(EV_DRAW, player, card(code, player == 0 || faceup(pos), player == 1 || faceup(pos)), {}, nullptr, &to,
                     drawn.size());
            if (player <= 1) hand_[player] += static_cast<int64_t>(drawn.size());
            break;
        }
        case MSG_MOVE: {
            const uint32_t code = r.u32();
            const Loc prev = r.loc_info(), cur = r.loc_info();
            const uint32_t reason = r.u32();
            const int owner = cur.location ? cur.controller : prev.controller;
            const bool pub = is_public(cur) || (prev.location != 0 && is_public(prev));
            emit(EV_MOVE, owner, card(code, owner == 0 || pub, owner == 1 || pub), {}, &prev, &cur, reason & 0x7FFFFFFF);
            int zone;
            if (slot_of(prev, zone)) {
                Slot& s = field_[prev.controller][zone][prev.sequence];
                if (s.used && (s.code == code || s.code == 0)) s = Slot{};  // 0: identity unknown
            }
            place(code, cur);
            hand_delta(prev, -1);
            hand_delta(cur, +1);
            auto link = links_.find(solving_);
            if (solving_ && link != links_.end() && prev.location == LOCATION_DECK) {
                if (cur.location == LOCATION_HAND) link->second.triggers |= kSearch;
                else if (cur.location == LOCATION_MZONE) link->second.triggers |= kSpsummonDeck;
                else if (cur.location == LOCATION_GRAVE) link->second.triggers |= kSendDeckGrave;
            }
            break;
        }
        case MSG_POS_CHANGE: {
            const uint32_t code = r.u32();
            const Loc loc = r.loc8();
            const uint8_t ppos = r.u8(), cpos = r.u8();
            const bool up = faceup(ppos) || faceup(cpos);
            const Loc from = make_loc(loc.controller, loc.location, loc.sequence, ppos);
            const Loc to = make_loc(loc.controller, loc.location, loc.sequence, cpos);
            emit(EV_POS_CHANGE, loc.controller, card(code, loc.controller == 0 || up, loc.controller == 1 || up), {}, &from,
                 &to);
            place(code, to);
            break;
        }
        case MSG_SET: {
            const uint32_t code = r.u32();
            const Loc loc = r.loc_info();
            emit(EV_SET, loc.controller, card(code, loc.controller == 0, loc.controller == 1), {}, nullptr, &loc);
            place(code, loc);
            break;
        }
        case MSG_SWAP: {
            const uint32_t code1 = r.u32();
            const Loc a = r.loc_info();
            const uint32_t code2 = r.u32();
            const Loc b = r.loc_info();
            const Card c1 = card(code1, b.controller == 0 || is_public(a), b.controller == 1 || is_public(a));
            const Card c2 = card(code2, a.controller == 0 || is_public(b), a.controller == 1 || is_public(b));
            emit(EV_SWAP, -1, c1, c2, &a, &b);
            place(code1, make_loc(b.controller, b.location, b.sequence, a.position));
            place(code2, make_loc(a.controller, a.location, a.sequence, b.position));
            break;
        }
        case MSG_SUMMONING:
        case MSG_SPSUMMONING:
        case MSG_FLIPSUMMONING: {
            const uint32_t code = r.u32();
            const Loc loc = r.loc_info();
            const int con = loc.controller;
            const bool pub = is_public(loc);
            const Card c = card(code, con == 0 || pub, con == 1 || pub);
            emit(type == MSG_SUMMONING ? EV_SUMMONING : type == MSG_SPSUMMONING ? EV_SPSUMMONING : EV_FLIPSUMMONING, con, c, {},
                 nullptr, &loc);
            place(code, loc);
            if (type != MSG_FLIPSUMMONING && con == turn_player_ && con <= 1) {
                if (++summons_ == 5) windows_.push_back(Window{1 - turn_player_, kFifthSummon, c, loc, false});
            }
            break;
        }
        case MSG_CHAINING: {
            const uint32_t code = r.u32();
            const Loc loc = r.loc_info();
            const uint8_t player = r.u8();
            r.u8();
            r.u32();
            const uint64_t desc = r.u64();
            const uint32_t count = r.u32();
            Card c2;
            int64_t v2 = 0, v3 = 0;
            if (desc) {
                const uint64_t owner = desc >> 20;
                if (owner <= 0xFFFFFFFFull && cards_->find(static_cast<uint32_t>(owner))) {
                    c2 = card(static_cast<uint32_t>(owner));
                    v2 = clamp(static_cast<int64_t>(desc & 0xFFFFF) + 1);
                } else {
                    v3 = static_cast<int64_t>(std::min<uint64_t>(desc, kClamp));
                }
            }
            emit(EV_CHAINING, player, card(code), c2, &loc, nullptr, clamp(count), v2, v3);
            links_.erase(links_.lower_bound(count), links_.end());
            links_[count] = Link{player, code, loc, 0};
            for (auto& w : windows_)
                if (w.abstainer == player) w.responded = true;
            break;
        }
        case MSG_CHAIN_SOLVING:
        case MSG_CHAIN_NEGATED:
        case MSG_CHAIN_DISABLED: {
            const uint8_t n = r.u8();
            auto it = links_.find(n);
            const bool known = it != links_.end();
            emit(type == MSG_CHAIN_SOLVING ? EV_CHAIN_SOLVING : type == MSG_CHAIN_NEGATED ? EV_CHAIN_NEGATED : EV_CHAIN_DISABLED,
                 known ? it->second.player : -1, known ? card(it->second.code) : Card{}, {}, nullptr, nullptr, n);
            if (type == MSG_CHAIN_SOLVING) solving_ = n;
            break;
        }
        case MSG_CHAIN_SOLVED: {
            const uint8_t n = r.u8();
            auto it = links_.find(n);
            if (it != links_.end() && it->second.player <= 1) {
                const int abstainer = 1 - it->second.player;
                bool responded = false;
                for (const auto& [k, link] : links_)
                    if (k > n && link.player == abstainer) responded = true;
                if (!responded)
                    abstain(abstainer, it->second.triggers ? it->second.triggers : kOther, card(it->second.code),
                            it->second.loc);
            }
            solving_ = 0;
            break;
        }
        case MSG_CHAIN_END:
            links_.clear();
            solving_ = 0;
            emit(EV_CHAIN_END);
            break;
        case MSG_NEW_TURN: {
            const uint8_t player = r.u8();
            close_windows();
            ++turn_;
            turn_player_ = player;
            summons_ = 0;
            emit(EV_NEW_TURN, player);
            break;
        }
        case MSG_NEW_PHASE: {
            const uint16_t phase = r.u16();
            close_windows();
            phase_ = phase;
            emit(EV_NEW_PHASE, -1, {}, {}, nullptr, nullptr, bit_index(phase));
            break;
        }
        case MSG_DAMAGE:
        case MSG_RECOVER:
        case MSG_PAY_LPCOST:
        case MSG_LPUPDATE: {
            const uint8_t player = r.u8();
            const uint32_t amount = r.u32();
            if (player <= 1) {
                if (type == MSG_RECOVER) lp_[player] += amount;
                else if (type == MSG_LPUPDATE) lp_[player] = amount;
                else lp_[player] -= amount;
            }
            const int kind = type == MSG_DAMAGE ? EV_DAMAGE : type == MSG_RECOVER ? EV_RECOVER
                             : type == MSG_PAY_LPCOST ? EV_PAY_LPCOST : EV_LP_UPDATE;
            emit(kind, player, {}, {}, nullptr, nullptr, clamp(amount));
            break;
        }
        case MSG_ATTACK: {
            const Loc attacker = r.loc_info(), target = r.loc_info();
            const Card c = field_card(attacker);
            const bool direct = target.location == 0;
            emit(EV_ATTACK, attacker.controller, c, direct ? Card{} : field_card(target), &attacker,
                 direct ? nullptr : &target, direct ? 1 : 0);
            if (attacker.controller <= 1) windows_.push_back(Window{1 - attacker.controller, kAttack, c, attacker, false});
            break;
        }
        case MSG_BATTLE: {
            const Loc attacker = r.loc_info();
            const int32_t atk = r.i32();
            r.i32();
            r.u8();
            const Loc target = r.loc_info();
            const int32_t tatk = r.i32(), tdef = r.i32();
            r.u8();
            emit(EV_BATTLE, attacker.controller, field_card(attacker), field_card(target), &attacker, &target, clamp(atk),
                 clamp(tatk), clamp(tdef));
            break;
        }
        case MSG_ATTACK_DISABLED:
            emit(EV_ATTACK_DISABLED);
            break;
        case MSG_EQUIP:
        case MSG_CARD_TARGET:
        case MSG_CANCEL_TARGET: {
            const Loc a = r.loc_info(), b = r.loc_info();
            emit(type == MSG_EQUIP ? EV_EQUIP : type == MSG_CARD_TARGET ? EV_CARD_TARGET : EV_CANCEL_TARGET, a.controller,
                 field_card(a), field_card(b), &a, &b);
            break;
        }
        case MSG_UNEQUIP: {
            const Loc a = r.loc_info();
            emit(EV_UNEQUIP, a.controller, field_card(a), {}, &a);
            break;
        }
        case MSG_BECOME_TARGET:
        case MSG_CARD_SELECTED:
        case MSG_RANDOM_SELECTED: {
            const int player = type == MSG_RANDOM_SELECTED ? r.u8() : -1;
            std::vector<Loc> locs;
            for (uint32_t n = r.u32(); n; --n) locs.push_back(r.loc_info());
            const int kind = type == MSG_BECOME_TARGET ? EV_BECOME_TARGET : type == MSG_CARD_SELECTED ? EV_CARD_SELECTED
                                                                                                   : EV_RANDOM_SELECTED;
            for (const auto& l : locs) emit(kind, player >= 0 ? player : l.controller, field_card(l), {}, &l);
            break;
        }
        case MSG_ADD_COUNTER:
        case MSG_REMOVE_COUNTER: {
            const uint16_t ctype = r.u16();
            const Loc l = r.loc8();
            const uint16_t count = r.u16();
            emit(type == MSG_ADD_COUNTER ? EV_ADD_COUNTER : EV_REMOVE_COUNTER, l.controller, field_card(l), {}, &l, nullptr,
                 ctype, count);
            break;
        }
        case MSG_CONFIRM_CARDS:
        case MSG_CONFIRM_DECKTOP:
        case MSG_CONFIRM_EXTRATOP: {
            const uint8_t player = r.u8();
            std::vector<std::pair<uint32_t, Loc>> shown;
            for (uint32_t n = r.u32(); n; --n) {
                const uint32_t code = r.u32();
                shown.emplace_back(code, r.loc32());
            }
            const int kind = type == MSG_CONFIRM_CARDS ? EV_CONFIRM_CARDS : type == MSG_CONFIRM_DECKTOP ? EV_CONFIRM_DECKTOP
                                                                                                     : EV_CONFIRM_EXTRATOP;
            for (const auto& [code, l] : shown) {
                const bool pub = type != MSG_CONFIRM_CARDS || l.location != LOCATION_DECK;
                emit(kind, player, card(code, player == 0 || pub, player == 1 || pub), {}, &l);
            }
            break;
        }
        case MSG_DECK_TOP: {
            const uint8_t player = r.u8();
            r.u32();
            const uint32_t code = r.u32(), position = r.u32();
            emit(EV_DECK_TOP, player, card(code, faceup(position), faceup(position)));
            break;
        }
        case MSG_SHUFFLE_DECK:
            emit(EV_SHUFFLE_DECK, r.u8());
            break;
        case MSG_SHUFFLE_HAND:
        case MSG_SHUFFLE_EXTRA: {
            const uint8_t player = r.u8();
            const uint32_t n = r.u32();
            for (uint32_t i = 0; i < n; ++i) r.u32();
            emit(type == MSG_SHUFFLE_HAND ? EV_SHUFFLE_HAND : EV_SHUFFLE_EXTRA, player, {}, {}, nullptr, nullptr, n);
            break;
        }
        case MSG_SHUFFLE_SET_CARD: {
            const uint8_t location = r.u8(), ct = r.u8();
            for (int i = 0; i < ct; ++i) {  // zones before the shuffle: still occupied, identity now unknown
                const Loc l = r.loc_info();
                int zone;
                if (slot_of(l, zone) && field_[l.controller][zone][l.sequence].used)
                    field_[l.controller][zone][l.sequence].code = 0;
            }
            for (int i = 0; i < ct; ++i) r.loc_info();
            const Loc from = make_loc(2, location, 0, 0);
            emit(EV_SHUFFLE_SET_CARD, -1, {}, {}, &from, nullptr, ct);
            break;
        }
        case MSG_SWAP_GRAVE_DECK: {
            const uint8_t player = r.u8();
            r.u32();
            r.skip(r.u32());
            emit(EV_SWAP_GRAVE_DECK, player);
            break;
        }
        case MSG_REVERSE_DECK:
            emit(EV_REVERSE_DECK);
            break;
        case MSG_FIELD_DISABLED: {
            const uint32_t flag = r.u32();
            const int64_t halves[2] = {flag & 0xFFFF, (flag >> 16) & 0xFFFF};
            for (int v = 0; v < 2; ++v) emit(EV_FIELD_DISABLED, -1, {}, {}, nullptr, nullptr, halves[v], halves[1 - v], 0, v);
            break;
        }
        case MSG_TOSS_COIN:
        case MSG_TOSS_DICE: {
            const uint8_t player = r.u8(), n = r.u8();
            int64_t packed = 0;
            for (int i = 0; i < n; ++i) {
                const uint8_t v = r.u8();
                if (type == MSG_TOSS_COIN && i < 30) packed |= static_cast<int64_t>(v & 1) << i;
                if (type == MSG_TOSS_DICE && i < 10) packed |= static_cast<int64_t>(v & 7) << (3 * i);
            }
            emit(type == MSG_TOSS_COIN ? EV_TOSS_COIN : EV_TOSS_DICE, player, {}, {}, nullptr, nullptr, n, packed);
            break;
        }
        case MSG_HAND_RES: {
            const uint8_t v = r.u8();
            const int64_t hands[2] = {v & 0x3, (v >> 2) & 0x3};
            for (int w = 0; w < 2; ++w) emit(EV_HAND_RES, -1, {}, {}, nullptr, nullptr, hands[w], hands[1 - w], 0, w);
            break;
        }
        default:
            break;
    }
}

}  // namespace ygorl::host
