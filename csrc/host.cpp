// See host.h. Every function here mirrors a Python reference; keep them in sync
// (tests/test_cpp_host.py fuzzes the decision states and runs real games in lockstep).
#include "host.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

#include "ocgapi_constants.h"

namespace ygorl::host {

namespace {

constexpr int kMaxConsecutiveRetries = 8;

struct DecodeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Reader {
public:
    Reader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}
    template <typename T>
    T get() {
        if (static_cast<size_t>(end_ - p_) < sizeof(T)) throw DecodeError("truncated payload");
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    uint8_t u8() { return get<uint8_t>(); }
    uint16_t u16() { return get<uint16_t>(); }
    uint32_t u32() { return get<uint32_t>(); }
    uint64_t u64() { return get<uint64_t>(); }
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
    CardRef card(Loc (Reader::*loc)()) {
        CardRef c;
        c.code = u32();
        c.loc = (this->*loc)();
        return c;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

std::string pack_i32(std::initializer_list<int32_t> values) {
    std::string out(values.size() * 4, '\0');
    size_t i = 0;
    for (int32_t v : values) std::memcpy(&out[4 * i++], &v, 4);
    return out;
}

template <typename T>
void append(std::string& out, T v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

bool contains(const std::vector<int64_t>& v, int64_t x) { return std::find(v.begin(), v.end(), x) != v.end(); }

Action make(Kind kind, int32_t index = -1) {
    Action a;
    a.kind = kind;
    a.index = index;
    return a;
}

Action with_card(Kind kind, int32_t index, const CardRef& card, uint64_t desc = 0, int64_t value = 0) {
    Action a = make(kind, index);
    a.has_card = true;
    a.card = card;
    a.description = desc;
    a.value = value;
    return a;
}

}  // namespace

// ------------------------------------------------------------------ decoding

bool is_decision_type(uint8_t t) {
    switch (t) {
        case MSG_SELECT_BATTLECMD: case MSG_SELECT_IDLECMD: case MSG_SELECT_EFFECTYN: case MSG_SELECT_YESNO:
        case MSG_SELECT_OPTION: case MSG_SELECT_CARD: case MSG_SELECT_CHAIN: case MSG_SELECT_PLACE:
        case MSG_SELECT_POSITION: case MSG_SELECT_TRIBUTE: case MSG_SORT_CHAIN: case MSG_SELECT_COUNTER:
        case MSG_SELECT_SUM: case MSG_SELECT_DISFIELD: case MSG_SORT_CARD: case MSG_SELECT_UNSELECT_CARD:
        case MSG_ROCK_PAPER_SCISSORS: case MSG_ANNOUNCE_RACE: case MSG_ANNOUNCE_ATTRIB: case MSG_ANNOUNCE_CARD:
        case MSG_ANNOUNCE_NUMBER:
            return true;
        default:
            return false;
    }
}

std::optional<Decision> decode_decision(const uint8_t* data, size_t size) {
    if (size < 1 || !is_decision_type(data[0])) return std::nullopt;
    Decision d;
    d.type = data[0];
    Reader r(data + 1, size - 1);
    try {
        d.player = r.u8();
        switch (d.type) {
            case MSG_SELECT_BATTLECMD: {
                for (uint32_t n = r.u32(); n; --n) {
                    ChainOption o;
                    o.card = r.card(&Reader::loc32);
                    o.description = r.u64();
                    r.u8();
                    d.activatable.push_back(o);
                }
                for (uint32_t n = r.u32(); n; --n) {
                    CardRef c = r.card(&Reader::loc8);
                    bool direct = r.u8() != 0;
                    d.attackable.emplace_back(c, direct);
                }
                d.can_main2 = r.u8() != 0;
                d.can_end_phase = r.u8() != 0;
                break;
            }
            case MSG_SELECT_IDLECMD: {
                for (auto* list : {&d.summonable, &d.spsummonable})
                    for (uint32_t n = r.u32(); n; --n) list->push_back(r.card(&Reader::loc32));
                for (uint32_t n = r.u32(); n; --n) d.repositionable.push_back(r.card(&Reader::loc8));
                for (auto* list : {&d.msetable, &d.ssetable})
                    for (uint32_t n = r.u32(); n; --n) list->push_back(r.card(&Reader::loc32));
                for (uint32_t n = r.u32(); n; --n) {
                    ChainOption o;
                    o.card = r.card(&Reader::loc32);
                    o.description = r.u64();
                    r.u8();
                    d.activatable.push_back(o);
                }
                d.can_battle_phase = r.u8() != 0;
                d.can_end_phase = r.u8() != 0;
                d.can_shuffle = r.u8() != 0;
                break;
            }
            case MSG_SELECT_EFFECTYN:
                d.has_card = true;
                d.card = r.card(&Reader::loc_info);
                d.description = r.u64();
                break;
            case MSG_SELECT_YESNO:
                d.description = r.u64();
                break;
            case MSG_SELECT_OPTION:
            case MSG_ANNOUNCE_NUMBER:
            case MSG_ANNOUNCE_CARD:
                for (uint8_t n = r.u8(); n; --n) d.options.push_back(r.u64());
                break;
            case MSG_SELECT_CARD:
                d.cancelable = r.u8() != 0;
                d.min = r.u32();
                d.max = r.u32();
                for (uint32_t n = r.u32(); n; --n) d.cards.push_back(r.card(&Reader::loc_info));
                break;
            case MSG_SELECT_UNSELECT_CARD:
                d.finishable = r.u8() != 0;
                d.cancelable = r.u8() != 0;
                d.min = r.u32();
                d.max = r.u32();
                for (uint32_t n = r.u32(); n; --n) d.cards.push_back(r.card(&Reader::loc_info));
                for (uint32_t n = r.u32(); n; --n) d.unselectable.push_back(r.card(&Reader::loc_info));
                break;
            case MSG_SELECT_CHAIN: {
                r.u8();  // special count
                d.forced = r.u8() != 0;
                r.u32();
                r.u32();
                for (uint32_t n = r.u32(); n; --n) {
                    ChainOption o;
                    o.card = r.card(&Reader::loc_info);
                    o.description = r.u64();
                    r.u8();
                    d.chains.push_back(o);
                }
                break;
            }
            case MSG_SELECT_PLACE:
            case MSG_SELECT_DISFIELD:
                d.place_count = r.u8();
                d.flag = r.u32();
                break;
            case MSG_SELECT_POSITION:
                d.position_code = r.u32();
                d.positions = r.u8();
                break;
            case MSG_SELECT_TRIBUTE:
                d.cancelable = r.u8() != 0;
                d.min = r.u32();
                d.max = r.u32();
                for (uint32_t n = r.u32(); n; --n) {
                    d.cards.push_back(r.card(&Reader::loc32));
                    d.params.push_back(r.u8());
                }
                break;
            case MSG_SELECT_COUNTER:
                d.counter_type = r.u16();
                d.count = r.u16();
                for (uint32_t n = r.u32(); n; --n) {
                    d.cards.push_back(r.card(&Reader::loc8));
                    d.params.push_back(r.u16());
                }
                break;
            case MSG_SELECT_SUM: {
                d.exact = r.u8() == 0;
                d.target = r.u32();
                d.min = r.u32();
                d.max = r.u32();
                for (uint32_t n = r.u32(); n; --n) {
                    r.card(&Reader::loc_info);
                    d.must_params.push_back(r.u32());
                }
                for (uint32_t n = r.u32(); n; --n) {
                    d.cards.push_back(r.card(&Reader::loc_info));
                    d.params.push_back(r.u32());
                }
                break;
            }
            case MSG_SORT_CARD:
            case MSG_SORT_CHAIN:
                for (uint32_t n = r.u32(); n; --n) {
                    CardRef c;
                    c.code = r.u32();
                    c.loc.controller = r.u8();
                    c.loc.location = static_cast<uint8_t>(r.u32());
                    c.loc.sequence = r.u32();
                    d.cards.push_back(c);
                }
                break;
            case MSG_ANNOUNCE_RACE:
                d.announce_count = r.u8();
                d.available = r.u64();
                break;
            case MSG_ANNOUNCE_ATTRIB:
                d.announce_count = r.u8();
                d.available = r.u32();
                break;
            case MSG_ROCK_PAPER_SCISSORS:
                break;
        }
    } catch (const DecodeError&) {
        return std::nullopt;
    }
    return d;
}

// ------------------------------------------------------------------ declarable

bool is_declarable(const CardRecord& rec, const std::vector<uint64_t>& opcodes) {
    const OCG_CardData& c = rec.data;
    std::vector<int64_t> st;
    bool alias = false, token = false;
    auto binary = [&](auto fn) {
        if (st.size() >= 2) {
            int64_t rhs = st.back();
            st.pop_back();
            int64_t lhs = st.back();
            st.pop_back();
            st.push_back(static_cast<int64_t>(fn(lhs, rhs)));
        }
    };
    auto unary = [&](auto fn) {
        if (!st.empty()) {
            int64_t a = st.back();
            st.pop_back();
            st.push_back(static_cast<int64_t>(fn(a)));
        }
    };
    for (uint64_t op : opcodes) {
        switch (op) {
            case OPCODE_ADD: binary([](int64_t a, int64_t b) { return a + b; }); break;
            case OPCODE_SUB: binary([](int64_t a, int64_t b) { return a - b; }); break;
            case OPCODE_MUL: binary([](int64_t a, int64_t b) { return a * b; }); break;
            case OPCODE_DIV: binary([](int64_t a, int64_t b) { return b ? a / b : 0; }); break;
            case OPCODE_AND: binary([](int64_t a, int64_t b) { return (a != 0) && (b != 0); }); break;
            case OPCODE_OR: binary([](int64_t a, int64_t b) { return (a != 0) || (b != 0); }); break;
            case OPCODE_NEG: unary([](int64_t a) { return -a; }); break;
            case OPCODE_NOT: unary([](int64_t a) { return a == 0; }); break;
            case OPCODE_BAND: binary([](int64_t a, int64_t b) { return a & b; }); break;
            case OPCODE_BOR: binary([](int64_t a, int64_t b) { return a | b; }); break;
            case OPCODE_BXOR: binary([](int64_t a, int64_t b) { return a ^ b; }); break;
            case OPCODE_BNOT: unary([](int64_t a) { return ~a; }); break;
            case OPCODE_LSHIFT: binary([](int64_t a, int64_t b) { return (b >= 0 && b < 64) ? (a << b) : 0; }); break;
            case OPCODE_RSHIFT: binary([](int64_t a, int64_t b) { return (b >= 0 && b < 64) ? (a >> b) : 0; }); break;
            case OPCODE_ISCODE: unary([&](int64_t a) { return c.code == static_cast<uint32_t>(a & 0xFFFFFFFF); }); break;
            case OPCODE_ISTYPE: unary([&](int64_t a) { return static_cast<int64_t>(c.type) & a; }); break;
            case OPCODE_ISRACE: unary([&](int64_t a) { return static_cast<int64_t>(c.race) & a; }); break;
            case OPCODE_ISATTRIBUTE: unary([&](int64_t a) { return static_cast<int64_t>(c.attribute) & a; }); break;
            case OPCODE_GETCODE: st.push_back(c.code); break;
            case OPCODE_GETTYPE: st.push_back(c.type); break;
            case OPCODE_GETRACE: st.push_back(static_cast<int64_t>(c.race)); break;
            case OPCODE_GETATTRIBUTE: st.push_back(c.attribute); break;
            case OPCODE_ISSETCARD:
                if (!st.empty()) {
                    uint32_t set_code = static_cast<uint32_t>(st.back() & 0xFFFFFFFF);
                    st.pop_back();
                    uint32_t settype = set_code & 0xFFF, subtype = set_code & 0xF000;
                    bool res = false;
                    for (uint16_t sc : rec.setcodes)
                        if (sc && (sc & 0xFFF) == settype && (sc & 0xF000 & subtype) == subtype) res = true;
                    st.push_back(res);
                }
                break;
            case OPCODE_ALLOW_ALIASES: alias = true; break;
            case OPCODE_ALLOW_TOKENS: token = true; break;
            default: st.push_back(static_cast<int64_t>(op)); break;
        }
    }
    if (st.size() != 1 || st[0] == 0) return false;
    if (c.code == 78734254 /*CARD_MARINE_DOLPHIN*/ || c.code == 13857930 /*CARD_TWINKLE_MOSS*/) return true;
    const uint32_t monster_token = TYPE_MONSTER | TYPE_TOKEN;
    return (alias || !c.alias) && (token || (c.type & monster_token) != monster_token);
}

// ------------------------------------------------------------------ decision states

DecisionState::DecisionState(Decision decision, const CardDatabase* cards) : d_(std::move(decision)), cards_(cards) {
    if (d_.type == MSG_ANNOUNCE_CARD && !cards_) throw std::invalid_argument("ANNOUNCE_CARD needs the card database");
}

const std::vector<Action>& DecisionState::actions() {
    static const std::vector<Action> none;
    if (done_) return none;
    if (!actions_valid_) {
        actions_ = legal();
        actions_valid_ = true;
    }
    return actions_;
}

bool DecisionState::step(size_t index) {
    if (done_) throw std::runtime_error("decision already complete");
    const auto& acts = actions();
    if (index >= acts.size())
        throw std::out_of_range("action " + std::to_string(index) + " out of range 0.." + std::to_string(acts.size()) + "-1");
    Action a = acts[index];
    actions_valid_ = false;
    done_ = apply(a);
    return done_;
}

void DecisionState::set_cards_response() {
    if (picked_.empty()) {
        response_ = pack_i32({-1});
        return;
    }
    response_.clear();
    append<int32_t>(response_, 0);
    append<uint32_t>(response_, static_cast<uint32_t>(picked_.size()));
    for (int64_t i : picked_) append<uint32_t>(response_, static_cast<uint32_t>(i));
}

bool DecisionState::tribute_can_reach(const std::vector<int>& chosen) const {
    int64_t sum = 0;
    for (int i : chosen) sum += d_.params[i];
    int slots = static_cast<int>(d_.max) - static_cast<int>(chosen.size());
    std::vector<uint32_t> rest;
    for (size_t i = 0; i < d_.params.size(); ++i)
        if (std::find(chosen.begin(), chosen.end(), static_cast<int>(i)) == chosen.end()) rest.push_back(d_.params[i]);
    std::sort(rest.begin(), rest.end(), std::greater<uint32_t>());
    for (int k = 0; k < std::max(slots, 0) && k < static_cast<int>(rest.size()); ++k) sum += rest[k];
    return sum >= static_cast<int64_t>(d_.min);
}

namespace {
std::vector<uint32_t> sum_values(uint32_t param) {
    uint32_t lo = param & 0xFFFF, hi = param >> 16;
    if (hi) return {lo, hi};
    return {lo};
}
std::set<int64_t> sums_of(const std::vector<uint32_t>& params, int64_t target) {
    std::set<int64_t> sums{0};
    for (uint32_t p : params) {
        std::set<int64_t> next;
        for (int64_t s : sums)
            for (uint32_t v : sum_values(p))
                if (s + v <= target) next.insert(s + v);
        sums.swap(next);
    }
    return sums;
}
std::pair<int64_t, int64_t> lo_hi(uint32_t param) {
    int64_t lo = param & 0xFFFF, hi = param >> 16;
    int64_t ms = (hi && hi < lo) ? hi : lo;
    return {ms, std::max(lo, hi)};
}
}  // namespace

bool DecisionState::sum_exact_feasible(const std::vector<int>& chosen) const {
    std::vector<uint32_t> base_params = d_.must_params;
    for (int i : chosen) base_params.push_back(d_.params[i]);
    const int64_t target = d_.target;
    const int max_count = static_cast<int>(std::max(d_.max, d_.min));
    if (static_cast<int>(chosen.size()) > max_count) return false;
    std::vector<std::set<int64_t>> dp(1 + max_count - chosen.size());
    dp[0] = sums_of(base_params, target);
    for (size_t i = 0; i < d_.params.size(); ++i) {
        if (std::find(chosen.begin(), chosen.end(), static_cast<int>(i)) != chosen.end()) continue;
        for (size_t k = dp.size() - 1; k >= 1; --k) {
            for (int64_t s : std::set<int64_t>(dp[k - 1]))
                for (uint32_t v : sum_values(d_.params[i]))
                    if (s + v <= target) dp[k].insert(s + v);
        }
    }
    for (size_t k = 0; k < dp.size(); ++k)
        if (chosen.size() + k >= d_.min && dp[k].count(target)) return true;
    return false;
}

bool DecisionState::sum_greater_valid(const std::vector<int>& chosen) const {
    std::vector<uint32_t> opts = d_.must_params;
    for (int i : chosen) opts.push_back(d_.params[i]);
    if (opts.empty()) return false;
    int64_t total_ms = 0, total_mx = 0, min_ms = INT64_MAX;
    for (uint32_t p : opts) {
        auto [ms, mx] = lo_hi(p);
        total_ms += ms;
        total_mx += mx;
        min_ms = std::min(min_ms, ms);
    }
    return total_mx >= static_cast<int64_t>(d_.target) && total_ms - min_ms < static_cast<int64_t>(d_.target);
}

bool DecisionState::sum_greater_dead(const std::vector<int>& chosen) const {
    std::vector<uint32_t> opts = d_.must_params;
    for (int i : chosen) opts.push_back(d_.params[i]);
    if (opts.empty()) return false;
    int64_t total_ms = 0, min_ms = INT64_MAX;
    for (uint32_t p : opts) {
        auto ms = lo_hi(p).first;
        total_ms += ms;
        min_ms = std::min(min_ms, ms);
    }
    return total_ms - min_ms >= static_cast<int64_t>(d_.target);
}

bool DecisionState::sum_feasible(const std::vector<int>& chosen) const {
    if (d_.exact) return sum_exact_feasible(chosen);
    std::map<std::vector<int>, bool> memo;
    std::function<bool(std::vector<int>)> search = [&](std::vector<int> state) -> bool {
        std::sort(state.begin(), state.end());
        auto it = memo.find(state);
        if (it != memo.end()) return it->second;
        bool result = false;
        if (sum_greater_dead(state)) {
            result = false;
        } else if (sum_greater_valid(state)) {
            result = true;
        } else {
            for (size_t i = 0; i < d_.params.size() && !result; ++i) {
                if (std::find(state.begin(), state.end(), static_cast<int>(i)) != state.end()) continue;
                std::vector<int> next = state;
                next.push_back(static_cast<int>(i));
                result = search(next);
            }
        }
        memo[state] = result;
        return result;
    };
    return search(chosen);
}

bool DecisionState::sum_complete(const std::vector<int>& chosen) const {
    if (!d_.exact) return sum_greater_valid(chosen);
    std::vector<uint32_t> params = d_.must_params;
    for (int i : chosen) params.push_back(d_.params[i]);
    return d_.min <= chosen.size() && chosen.size() <= std::max(d_.max, d_.min) &&
           sums_of(params, d_.target).count(d_.target);
}

std::vector<uint32_t> DecisionState::place_zones() const {
    std::vector<uint32_t> zones;
    const std::pair<int, uint32_t> sides[2] = {{0, d_.player}, {16, 1u - d_.player}};
    for (auto [rel, player] : sides) {
        for (uint32_t seq = 0; seq < 7; ++seq)
            if (!((d_.flag >> (rel + seq)) & 1)) zones.push_back(player << 16 | LOCATION_MZONE << 8 | seq);
        for (uint32_t seq = 0; seq < 8; ++seq)
            if (!((d_.flag >> (rel + 8 + seq)) & 1)) zones.push_back(player << 16 | LOCATION_SZONE << 8 | seq);
    }
    return zones;
}

std::vector<Action> DecisionState::legal() const {
    std::vector<Action> acts;
    auto picked_int = [&] {
        std::vector<int> v;
        for (int64_t p : picked_) v.push_back(static_cast<int>(p));
        return v;
    };
    switch (d_.type) {
        case MSG_SELECT_IDLECMD: {
            const std::pair<Kind, const std::vector<CardRef>*> lists[] = {
                {SUMMON, &d_.summonable}, {SPSUMMON, &d_.spsummonable}, {REPOSITION, &d_.repositionable},
                {MSET, &d_.msetable}, {SSET, &d_.ssetable}};
            for (auto [kind, list] : lists)
                for (size_t i = 0; i < list->size(); ++i) acts.push_back(with_card(kind, i, (*list)[i]));
            for (size_t i = 0; i < d_.activatable.size(); ++i)
                acts.push_back(with_card(ACTIVATE, i, d_.activatable[i].card, d_.activatable[i].description));
            if (d_.can_battle_phase) acts.push_back(make(BATTLE_PHASE));
            if (d_.can_end_phase) acts.push_back(make(END_PHASE));
            if (d_.can_shuffle) acts.push_back(make(SHUFFLE));
            break;
        }
        case MSG_SELECT_BATTLECMD:
            for (size_t i = 0; i < d_.activatable.size(); ++i)
                acts.push_back(with_card(ACTIVATE, i, d_.activatable[i].card, d_.activatable[i].description));
            for (size_t i = 0; i < d_.attackable.size(); ++i)
                acts.push_back(with_card(ATTACK, i, d_.attackable[i].first, 0, d_.attackable[i].second ? 1 : 0));
            if (d_.can_main2) acts.push_back(make(MAIN2));
            if (d_.can_end_phase) acts.push_back(make(END_PHASE));
            break;
        case MSG_SELECT_EFFECTYN:
        case MSG_SELECT_YESNO: {
            for (auto [kind, value] : {std::pair<Kind, int>{YES, 1}, {NO, 0}}) {
                Action a = make(kind);
                a.has_card = d_.has_card;
                a.card = d_.card;
                a.description = d_.description;
                a.value = value;
                acts.push_back(a);
            }
            break;
        }
        case MSG_SELECT_OPTION:
            for (size_t i = 0; i < d_.options.size(); ++i) {
                Action a = make(OPTION, i);
                a.description = d_.options[i];
                a.value = i;
                acts.push_back(a);
            }
            break;
        case MSG_ANNOUNCE_NUMBER:
            for (size_t i = 0; i < d_.options.size(); ++i) {
                Action a = make(NUMBER, i);
                a.value = static_cast<int64_t>(d_.options[i]);
                acts.push_back(a);
            }
            break;
        case MSG_ROCK_PAPER_SCISSORS:
            for (int i = 0; i < 3; ++i) {
                Action a = make(RPS, i);
                a.value = i + 1;
                acts.push_back(a);
            }
            break;
        case MSG_SELECT_CHAIN:
            for (size_t i = 0; i < d_.chains.size(); ++i)
                acts.push_back(with_card(CHAIN, i, d_.chains[i].card, d_.chains[i].description, i));
            if (!d_.forced) {
                Action a = make(PASS);
                a.value = -1;
                acts.push_back(a);
            }
            break;
        case MSG_SELECT_POSITION:
            for (uint32_t p : {POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE})
                if (d_.positions & p) {
                    CardRef c;
                    c.code = d_.position_code;
                    c.loc.controller = d_.player;
                    acts.push_back(with_card(POSITION, -1, c, 0, p));
                }
            break;
        case MSG_SELECT_UNSELECT_CARD: {
            const size_t n = d_.cards.size();
            for (size_t i = 0; i < n; ++i) acts.push_back(with_card(SELECT, i, d_.cards[i], 0, i));
            for (size_t i = 0; i < d_.unselectable.size(); ++i)
                acts.push_back(with_card(UNSELECT, i, d_.unselectable[i], 0, n + i));
            if (d_.finishable) {
                Action a = make(FINISH);
                a.value = -1;
                acts.push_back(a);
            } else if (d_.cancelable) {
                Action a = make(CANCEL);
                a.value = -1;
                acts.push_back(a);
            }
            break;
        }
        case MSG_ANNOUNCE_CARD: {
            std::vector<uint32_t> codes;
            for (const auto& [code, rec] : cards_->all())
                if (is_declarable(rec, d_.options)) codes.push_back(code);
            std::sort(codes.begin(), codes.end());
            for (uint32_t code : codes) {
                Action a = make(DECLARE);
                a.value = code;
                acts.push_back(a);
            }
            break;
        }
        case MSG_SELECT_CARD: {
            const size_t n = picked_.size();
            if (n < d_.max)
                for (size_t i = 0; i < d_.cards.size(); ++i)
                    if (!contains(picked_, i)) acts.push_back(with_card(SELECT, i, d_.cards[i]));
            if (n >= d_.min && (n > 0 || d_.cancelable))
                acts.push_back(make(FINISH));
            else if (d_.cancelable && n == 0)
                acts.push_back(make(CANCEL));
            break;
        }
        case MSG_SELECT_TRIBUTE: {
            auto chosen = picked_int();
            if (picked_.size() < d_.max)
                for (size_t i = 0; i < d_.cards.size(); ++i) {
                    if (contains(picked_, i)) continue;
                    auto next = chosen;
                    next.push_back(static_cast<int>(i));
                    if (tribute_can_reach(next)) acts.push_back(with_card(SELECT, i, d_.cards[i], 0, d_.params[i]));
                }
            int64_t sum = 0;
            for (int i : chosen) sum += d_.params[i];
            bool enough = sum >= static_cast<int64_t>(d_.min);
            if (enough && (!picked_.empty() || d_.cancelable))
                acts.push_back(make(FINISH));
            else if (d_.cancelable && picked_.empty())
                acts.push_back(make(CANCEL));
            break;
        }
        case MSG_SELECT_SUM: {
            auto chosen = picked_int();
            for (size_t i = 0; i < d_.cards.size(); ++i) {
                if (contains(picked_, i)) continue;
                auto next = chosen;
                next.push_back(static_cast<int>(i));
                if (sum_feasible(next)) acts.push_back(with_card(SELECT, i, d_.cards[i], 0, d_.params[i]));
            }
            if (!picked_.empty() && sum_complete(chosen)) acts.push_back(make(FINISH));
            break;
        }
        case MSG_SELECT_COUNTER:
            for (size_t i = 0; i < d_.cards.size(); ++i) {
                int64_t remaining = static_cast<int64_t>(d_.params[i]) - std::count(picked_.begin(), picked_.end(), static_cast<int64_t>(i));
                if (remaining > 0) acts.push_back(with_card(COUNTER, i, d_.cards[i], 0, remaining));
            }
            break;
        case MSG_SORT_CARD:
        case MSG_SORT_CHAIN:
            for (size_t i = 0; i < d_.cards.size(); ++i)
                if (!contains(picked_, i)) acts.push_back(with_card(SORT, i, d_.cards[i]));
            if (picked_.empty()) acts.push_back(make(DEFAULT));
            break;
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD:
            for (uint32_t z : place_zones())
                if (!contains(picked_, z)) {
                    Action a = make(PLACE);
                    a.value = z;
                    acts.push_back(a);
                }
            break;
        case MSG_ANNOUNCE_RACE:
        case MSG_ANNOUNCE_ATTRIB:
            for (int i = 0; i < 64; ++i)
                if ((d_.available >> i) & 1) {
                    int64_t bit = static_cast<int64_t>(uint64_t{1} << i);
                    if (!contains(picked_, bit)) {
                        Action a = make(d_.type == MSG_ANNOUNCE_RACE ? RACE : ATTRIBUTE);
                        a.value = bit;
                        acts.push_back(a);
                    }
                }
            break;
    }
    return acts;
}

bool DecisionState::apply(const Action& a) {
    switch (d_.type) {
        case MSG_SELECT_IDLECMD: {
            static const int codes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};  // kind order SUMMON..SHUFFLE
            response_ = pack_i32({codes[a.kind] | (std::max(a.index, 0) << 16)});
            return true;
        }
        case MSG_SELECT_BATTLECMD: {
            int code = a.kind == ACTIVATE ? 0 : a.kind == ATTACK ? 1 : a.kind == MAIN2 ? 2 : 3;
            response_ = pack_i32({code | (std::max(a.index, 0) << 16)});
            return true;
        }
        case MSG_SELECT_EFFECTYN: case MSG_SELECT_YESNO: case MSG_SELECT_OPTION: case MSG_ROCK_PAPER_SCISSORS:
        case MSG_SELECT_CHAIN: case MSG_SELECT_POSITION: case MSG_ANNOUNCE_CARD:
            response_ = pack_i32({static_cast<int32_t>(a.value)});
            return true;
        case MSG_ANNOUNCE_NUMBER:
            response_ = pack_i32({a.index});
            return true;
        case MSG_SELECT_UNSELECT_CARD:
            response_ = a.value == -1 ? pack_i32({-1}) : pack_i32({1, static_cast<int32_t>(a.value)});
            return true;
        case MSG_SELECT_CARD:
            if (a.kind == CANCEL) {
                response_ = pack_i32({-1});
                return true;
            }
            if (a.kind == SELECT) {
                picked_.push_back(a.index);
                if (picked_.size() < d_.max && picked_.size() < d_.cards.size()) return false;
            }
            set_cards_response();
            return true;
        case MSG_SELECT_TRIBUTE:
            if (a.kind == CANCEL) {
                response_ = pack_i32({-1});
                return true;
            }
            if (a.kind == SELECT) {
                picked_.push_back(a.index);
                if (!legal().empty() && picked_.size() < d_.max) return false;
            }
            set_cards_response();
            return true;
        case MSG_SELECT_SUM:
            if (a.kind == SELECT) {
                picked_.push_back(a.index);
                std::vector<int> chosen(picked_.begin(), picked_.end());
                if (!sum_complete(chosen)) return false;
                for (const Action& next : legal())
                    if (next.kind == SELECT) return false;
            }
            set_cards_response();
            return true;
        case MSG_SELECT_COUNTER: {
            picked_.push_back(a.index);
            if (picked_.size() < d_.count) return false;
            response_.clear();
            for (size_t i = 0; i < d_.cards.size(); ++i)
                append<int16_t>(response_, static_cast<int16_t>(std::count(picked_.begin(), picked_.end(), static_cast<int64_t>(i))));
            return true;
        }
        case MSG_SORT_CARD:
        case MSG_SORT_CHAIN: {
            if (a.kind == DEFAULT) {
                response_ = std::string(1, static_cast<char>(-1));
                return true;
            }
            picked_.push_back(a.index);
            const size_t n = d_.cards.size();
            if (picked_.size() + 1 < n) return false;
            for (size_t i = 0; i < n; ++i)
                if (!contains(picked_, i)) picked_.push_back(i);
            response_.clear();
            for (size_t i = 0; i < n; ++i)
                response_.push_back(static_cast<char>(std::find(picked_.begin(), picked_.end(), static_cast<int64_t>(i)) - picked_.begin()));
            return true;
        }
        case MSG_SELECT_PLACE:
        case MSG_SELECT_DISFIELD: {
            picked_.push_back(a.value);
            if (picked_.size() < d_.place_count) return false;
            response_.clear();
            for (int64_t z : picked_) {
                response_.push_back(static_cast<char>(z >> 16));
                response_.push_back(static_cast<char>((z >> 8) & 0xFF));
                response_.push_back(static_cast<char>(z & 0xFF));
            }
            return true;
        }
        case MSG_ANNOUNCE_RACE:
        case MSG_ANNOUNCE_ATTRIB: {
            picked_.push_back(a.value);
            if (picked_.size() < d_.announce_count) return false;
            uint64_t mask = 0;
            for (int64_t b : picked_) mask |= static_cast<uint64_t>(b);
            response_.clear();
            if (d_.type == MSG_ANNOUNCE_RACE)
                append<uint64_t>(response_, mask);
            else
                append<uint32_t>(response_, static_cast<uint32_t>(mask));
            return true;
        }
    }
    throw std::logic_error("unhandled decision type");
}

// ------------------------------------------------------------------ tracker

Tracker::Tracker(TrackerConfig cfg, const CardDatabase* cards) : cfg_(cfg), cards_(cards) {
    lp_ = {cfg.starting_lp, cfg.starting_lp};
}

void Tracker::stop(const std::string& reason, const std::string& error) {
    reason_ = reason;
    if (!error.empty()) error_ = error;
    done_ = true;
    state_.reset();
}

void Tracker::on_buffer(const std::string& buf, int status) {
    state_.reset();
    std::optional<Decision> decision;
    bool retried = false;
    size_t pos = 0;
    const auto* data = reinterpret_cast<const uint8_t*>(buf.data());
    while (pos + 4 <= buf.size()) {
        uint32_t len;
        std::memcpy(&len, data + pos, 4);
        if (len == 0 || pos + 4 + len > buf.size()) break;  // corrupt framing: ignore the rest
        const uint8_t* rec = data + pos + 4;
        const uint8_t type = rec[0];
        const size_t plen = len - 1;
        auto u32_at = [&](size_t off) {
            uint32_t v;
            std::memcpy(&v, rec + 1 + off, 4);
            return v;
        };
        if (is_decision_type(type)) {
            if (auto d = decode_decision(rec, len)) decision = std::move(d);
        } else {
            switch (type) {
                case MSG_NEW_TURN:
                    if (plen >= 1) {
                        ++turn_;
                        turn_player_ = rec[1];
                    }
                    break;
                case MSG_NEW_PHASE:
                    if (plen >= 2) std::memcpy(&phase_, rec + 1, 2);
                    break;
                case MSG_DAMAGE:
                case MSG_PAY_LPCOST:
                    if (plen >= 5 && rec[1] < 2) lp_[rec[1]] -= u32_at(1);
                    break;
                case MSG_RECOVER:
                    if (plen >= 5 && rec[1] < 2) lp_[rec[1]] += u32_at(1);
                    break;
                case MSG_LPUPDATE:
                    if (plen >= 5 && rec[1] < 2) lp_[rec[1]] = u32_at(1);
                    break;
                case MSG_WIN:
                    if (plen >= 2) {
                        engine_winner_ = rec[1] < 2 ? rec[1] : -1;
                        win_reason_ = rec[2];
                        reason_ = "win";
                    }
                    break;
                case MSG_RETRY:
                    retried = true;
                    ++retries_;
                    break;
                case MSG_START: case MSG_UPDATE_DATA: case MSG_UPDATE_CARD: case MSG_REQUEST_DECK:
                case MSG_REFRESH_DECK: case MSG_BE_CHAIN_TARGET: case MSG_CREATE_RELATION:
                case MSG_RELEASE_RELATION: case MSG_CUSTOM_MSG:
                    ++unknown_;  // not emitted by the core: decoded as UnknownMessage in Python
                    break;
                default:
                    break;
            }
        }
        pos += 4 + len;
    }
    if (reason_ == "win") {
        done_ = true;
        return;
    }
    if (status == OCG_DUEL_STATUS_END) {
        stop("end");
        return;
    }
    if (status != OCG_DUEL_STATUS_AWAITING) return;
    if (!decision && retried) {
        if (++consecutive_retries_ > kMaxConsecutiveRetries) {
            stop("error", "response rejected repeatedly (MSG_RETRY)");
            return;
        }
        decision = last_decision_;
    } else {
        consecutive_retries_ = 0;
    }
    if (!decision) {
        stop("error", "engine awaits a response but sent no decodable decision");
        return;
    }
    last_decision_ = decision;
    if (turn_ > cfg_.max_turns) {
        stop("turn_limit");
        return;
    }
    state_ = std::make_unique<DecisionState>(*decision, cards_);
}

const std::vector<Action>* Tracker::actions() {
    if (!awaiting()) return nullptr;
    if (decisions_ >= cfg_.max_decisions) {
        stop("decision_limit");
        return nullptr;
    }
    const auto& acts = state_->actions();
    if (acts.empty()) {
        stop("error", "no legal action");
        return nullptr;
    }
    return &acts;
}

const std::string* Tracker::act(size_t index) {
    if (!actions()) throw std::runtime_error("no decision is pending");
    state_->step(index);  // throws out_of_range before any bookkeeping
    ++decisions_;
    if (state_->done()) {
        responses_.push_back(state_->response());
        state_.reset();
        return &responses_.back();
    }
    return nullptr;
}

int Tracker::winner() const {
    if (reason_ == "turn_limit" || reason_ == "decision_limit" || reason_ == "error") {
        if (reason_ == "error" || lp_[0] == lp_[1]) return -1;
        return lp_[0] > lp_[1] ? 0 : 1;
    }
    return engine_winner_;
}

// ------------------------------------------------------------------ HostDuel

HostDuel::HostDuel(std::shared_ptr<CardDatabase> cards, std::shared_ptr<ScriptSource> scripts,
                   std::shared_ptr<const Vocab> vocab)
    : cards_(std::move(cards)), scripts_(std::move(scripts)), vocab_(std::move(vocab)) {}

void HostDuel::start(const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
                     const PlayerOptions& team2,
                     const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>& decks,
                     uint32_t max_turns, uint32_t max_decisions) {
    core_ = std::make_unique<Duel>(seed, flags, team1, team2, cards_, scripts_);
    for (const char* base : {"constant.lua", "utility.lua"})
        if (!core_->load_script(base)) throw std::runtime_error(std::string("failed to load base script ") + base);
    for (uint8_t team = 0; team < 2 && team < decks.size(); ++team) {
        for (uint32_t code : decks[team].first) core_->new_card(team, 0, code, team, LOCATION_DECK, 0, POS_FACEDOWN_DEFENSE);
        for (uint32_t code : decks[team].second) core_->new_card(team, 0, code, team, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
    }
    core_->start();
    tracker_ = std::make_unique<Tracker>(TrackerConfig{team1.starting_lp, max_turns, max_decisions}, cards_.get());
    advance();
}

void HostDuel::advance() {
    while (true) {
        int status = core_->process();
        std::string buf = core_->get_message();
        core_->pop_logs();
        tracker_->on_buffer(buf, status);
        if (tracker_->done() || tracker_->awaiting()) return;
    }
}

const std::vector<Action>& HostDuel::actions() {
    if (!tracker_) return empty_;
    const auto* acts = tracker_->actions();
    return acts ? *acts : empty_;
}

int HostDuel::player() const {
    const Decision* d = tracker_ ? tracker_->decision() : nullptr;
    return d ? d->player : -1;
}

void HostDuel::act(size_t index) {
    const std::string* response = tracker_->act(index);
    if (response) {
        core_->set_response(*response);
        advance();
    }
}

void HostDuel::observe(Observation& out) {
    encode(*core_, *tracker_, actions(), *cards_, *vocab_, out);
}

}  // namespace ygorl::host
