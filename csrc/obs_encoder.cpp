// Observation encoder: C++ mirror of ygorl/env/encoding.py (spec: docs/encoding.md).
#include <algorithm>
#include <cstring>
#include <map>
#include <tuple>

#include "host.h"
#include "ocgapi_constants.h"

namespace ygorl::host {

namespace {

constexpr uint32_t kCardQueryFlags = QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK |
                                     QUERY_ATTRIBUTE | QUERY_RACE | QUERY_ATTACK | QUERY_DEFENSE | QUERY_OVERLAY_CARD |
                                     QUERY_COUNTERS | QUERY_OWNER | QUERY_STATUS | QUERY_LSCALE | QUERY_RSCALE | QUERY_LINK;
constexpr int64_t kClamp = 65535;

namespace col {
enum Col { CARD_INDEX, LOCATION, SEQUENCE, OVERLAY_INDEX, CONTROLLER, OWNER, POSITION, VISIBLE, PUBLIC, TYPE,
           ATTRIBUTE, RACE, LEVEL, RANK, LINK, LSCALE, RSCALE, ATTACK, DEFENSE, LINK_MARKER, COUNTERS, MATERIALS,
           DISABLED };
}  // namespace col

struct QCard {
    uint32_t code = 0, position = 0, type = 0, level = 0, rank = 0, attribute = 0, status = 0, lscale = 0, rscale = 0;
    uint32_t link = 0, link_marker = 0;
    uint64_t race = 0;
    int32_t attack = 0, defense = 0;
    bool has_owner = false;
    uint8_t owner = 0, is_public = 0;
    std::vector<uint32_t> overlay, counters;
};

template <typename T>
T rd(const std::string& b, size_t pos) {
    T v;
    std::memcpy(&v, b.data() + pos, sizeof(T));
    return v;
}

std::vector<std::optional<QCard>> parse_location(const std::string& buf) {
    std::vector<std::optional<QCard>> out;
    if (buf.size() < 4) return out;
    size_t pos = 4;
    while (pos < buf.size()) {
        uint16_t len = rd<uint16_t>(buf, pos);
        if (len == 0) {
            out.emplace_back(std::nullopt);
            pos += 2;
            continue;
        }
        QCard c;
        while (true) {
            len = rd<uint16_t>(buf, pos);
            uint32_t flag = rd<uint32_t>(buf, pos + 2);
            size_t body = pos + 6;
            pos += 2 + len;
            if (flag == QUERY_END) break;
            switch (flag) {
                case QUERY_CODE: c.code = rd<uint32_t>(buf, body); break;
                case QUERY_POSITION: c.position = rd<uint32_t>(buf, body); break;
                case QUERY_TYPE: c.type = rd<uint32_t>(buf, body); break;
                case QUERY_LEVEL: c.level = rd<uint32_t>(buf, body); break;
                case QUERY_RANK: c.rank = rd<uint32_t>(buf, body); break;
                case QUERY_ATTRIBUTE: c.attribute = rd<uint32_t>(buf, body); break;
                case QUERY_RACE: c.race = rd<uint64_t>(buf, body); break;
                case QUERY_ATTACK: c.attack = rd<int32_t>(buf, body); break;
                case QUERY_DEFENSE: c.defense = rd<int32_t>(buf, body); break;
                case QUERY_STATUS: c.status = rd<uint32_t>(buf, body); break;
                case QUERY_LSCALE: c.lscale = rd<uint32_t>(buf, body); break;
                case QUERY_RSCALE: c.rscale = rd<uint32_t>(buf, body); break;
                case QUERY_OWNER: c.owner = rd<uint8_t>(buf, body); c.has_owner = true; break;
                case QUERY_IS_PUBLIC: c.is_public = rd<uint8_t>(buf, body); break;
                case QUERY_LINK:
                    c.link = rd<uint32_t>(buf, body);
                    c.link_marker = rd<uint32_t>(buf, body + 4);
                    break;
                case QUERY_OVERLAY_CARD:
                case QUERY_COUNTERS: {
                    uint32_t n = rd<uint32_t>(buf, body);
                    auto& dst = flag == QUERY_OVERLAY_CARD ? c.overlay : c.counters;
                    for (uint32_t i = 0; i < n; ++i) dst.push_back(rd<uint32_t>(buf, body + 4 + 4 * i));
                    break;
                }
                default: break;
            }
        }
        out.emplace_back(std::move(c));
    }
    return out;
}

int32_t bit_index(uint64_t v) {
    if (!v) return 0;
    int32_t i = 1;
    while (!(v & 1)) {
        v >>= 1;
        ++i;
    }
    return i;
}

int32_t clamp(int64_t v, int64_t lo = 0, int64_t hi = kClamp) { return static_cast<int32_t>(std::max(lo, std::min(hi, v))); }

int32_t location_enum(uint32_t loc) {
    switch (loc) {
        case LOCATION_DECK: return 1;
        case LOCATION_HAND: return 2;
        case LOCATION_MZONE: return 3;
        case LOCATION_SZONE: return 4;
        case LOCATION_GRAVE: return 5;
        case LOCATION_REMOVED: return 6;
        case LOCATION_EXTRA: return 7;
        case LOCATION_OVERLAY: return 8;
        default: return 0;
    }
}

int32_t position_enum(uint32_t pos) {
    switch (pos) {
        case POS_FACEUP_ATTACK: return 1;
        case POS_FACEDOWN_ATTACK: return 2;
        case POS_FACEUP_DEFENSE: return 3;
        case POS_FACEDOWN_DEFENSE: return 4;
        default: return 0;
    }
}

using Row = std::array<int32_t, F_CARD>;

Row card_row(const QCard& c, uint32_t loc, uint32_t seq, int side, int viewer, bool visible, const Vocab& vocab) {
    Row r{};
    r[col::LOCATION] = location_enum(loc);
    r[col::SEQUENCE] = static_cast<int32_t>(seq);
    r[col::CONTROLLER] = side;
    r[col::OWNER] = (c.has_owner ? c.owner : viewer) == viewer ? 0 : 1;
    r[col::POSITION] = position_enum(c.position);
    if (!visible) {
        r[col::CARD_INDEX] = Vocab::UNKNOWN;
        return r;
    }
    r[col::CARD_INDEX] = vocab.index(c.code);
    r[col::VISIBLE] = 1;
    r[col::PUBLIC] = c.is_public ? 1 : 0;
    r[col::TYPE] = static_cast<int32_t>(c.type & 0x7FFFFFFF);
    r[col::ATTRIBUTE] = bit_index(c.attribute);
    r[col::RACE] = bit_index(c.race);
    r[col::LEVEL] = static_cast<int32_t>(c.level);
    r[col::RANK] = static_cast<int32_t>(c.rank);
    r[col::LINK] = static_cast<int32_t>(c.link);
    r[col::LSCALE] = static_cast<int32_t>(c.lscale);
    r[col::RSCALE] = static_cast<int32_t>(c.rscale);
    r[col::ATTACK] = clamp(c.attack);
    r[col::DEFENSE] = clamp(c.defense);
    r[col::LINK_MARKER] = static_cast<int32_t>(c.link_marker);
    int64_t counters = 0;
    for (uint32_t v : c.counters) counters += v >> 16;
    r[col::COUNTERS] = static_cast<int32_t>(counters);
    r[col::MATERIALS] = static_cast<int32_t>(c.overlay.size());
    r[col::DISABLED] = (c.status & STATUS_DISABLED) ? 1 : 0;
    return r;
}

Row material_row(uint32_t code, uint32_t seq, uint32_t k, int side, const CardDatabase& cards, const Vocab& vocab) {
    Row r{};
    r[col::CARD_INDEX] = vocab.index(code);
    r[col::LOCATION] = location_enum(LOCATION_OVERLAY);
    r[col::SEQUENCE] = static_cast<int32_t>(seq);
    r[col::OVERLAY_INDEX] = static_cast<int32_t>(k + 1);
    r[col::CONTROLLER] = side;
    r[col::OWNER] = side;
    r[col::VISIBLE] = 1;
    r[col::PUBLIC] = 1;
    if (const CardRecord* rec = cards.find(code)) {
        const OCG_CardData& d = rec->data;
        const bool xyz = d.type & TYPE_XYZ, link = d.type & TYPE_LINK;
        r[col::TYPE] = static_cast<int32_t>(d.type & 0x7FFFFFFF);
        r[col::ATTRIBUTE] = bit_index(d.attribute);
        r[col::RACE] = bit_index(d.race);
        r[col::LEVEL] = (xyz || link) ? 0 : static_cast<int32_t>(d.level);
        r[col::RANK] = xyz ? static_cast<int32_t>(d.level) : 0;
        r[col::LINK] = link ? static_cast<int32_t>(d.level) : 0;
        r[col::LSCALE] = static_cast<int32_t>(d.lscale);
        r[col::RSCALE] = static_cast<int32_t>(d.rscale);
        r[col::ATTACK] = clamp(d.attack);
        r[col::DEFENSE] = clamp(d.defense);
        r[col::LINK_MARKER] = static_cast<int32_t>(d.link_marker);
    }
    return r;
}

// Keep only the first row of each class of equivalent action rows in the mask (docs/encoding.md 「等价动作去重」;
// Python: ygorl.env.encoding.mask_duplicates). Rows without a card-table row or with a hidden card are never merged.
void mask_duplicates(const std::vector<int32_t>& cards, const std::vector<int32_t>& actions,
                     std::vector<int32_t>& mask, uint32_t decision) {
    // deck, hand, Extra Deck: the sequence carries nothing a choice between copies could use (GY / banished
    // order is age, which tells apart per-card state such as "sent to the GY this turn")
    auto unordered = [](int32_t loc) { return loc == 1 || loc == 2 || loc == 7; };
    std::map<std::vector<int32_t>, size_t> first;
    const size_t n = static_cast<size_t>(std::count(mask.begin(), mask.end(), 1));  // legal rows are a prefix here
    for (size_t i = 0; i < n; ++i) {
        const int32_t* a = actions.data() + i * A_ACTION;
        const int32_t card_row = a[1], card_index = a[2];
        if (card_row == 0 || card_index == 0) continue;
        std::vector<int32_t> key(a, a + A_ACTION);
        key[1] = key[9] = 0;
        if (decision == MSG_SELECT_UNSELECT_CARD) key[8] = 0;  // the list index there
        const int32_t* c = cards.data() + static_cast<size_t>(card_row - 1) * F_CARD;
        if (!c[col::VISIBLE]) continue;  // never let the mask say that two hidden cards are the same
        key.insert(key.end(), c, c + F_CARD);
        int32_t* kc = key.data() + A_ACTION;
        kc[col::OVERLAY_INDEX] = 0;
        if (unordered(kc[col::LOCATION])) kc[col::SEQUENCE] = 0;
        if (!first.emplace(std::move(key), i).second) mask[i] = 0;
    }
}

uint32_t chain_size(const std::string& f) {
    size_t pos = 4;  // duel options
    for (int p = 0; p < 2; ++p) {
        pos += 4;  // lp
        for (int slot = 0; slot < 15; ++slot) {  // 7 monster + 8 spell/trap zones
            uint8_t present = rd<uint8_t>(f, pos);
            pos += 1;
            if (present) pos += 1 + 4;
        }
        pos += 6 * 4;
    }
    return rd<uint32_t>(f, pos);
}

}  // namespace

Vocab::Vocab(const std::vector<uint32_t>& passwords) {
    for (size_t i = 0; i < passwords.size(); ++i) index_.emplace(passwords[i], static_cast<int32_t>(i) + FIRST_INDEX);
}

int32_t Vocab::index(uint32_t password) const {
    auto it = index_.find(password);
    return it == index_.end() ? UNKNOWN : it->second;
}

void encode(Duel& core, const Tracker& tracker, const std::vector<Action>& actions, const CardDatabase& cards,
            const Vocab& vocab, Observation& out) {
    const Decision* decision = tracker.decision();
    const int viewer = decision ? decision->player : 0;
    std::vector<Row> rows;
    std::map<std::tuple<int, int, int64_t, int64_t>, size_t> keys;  // (con, loc, seq, overlay index or -1)
    std::map<uint32_t, size_t> deck_rows;

    static const uint32_t side_locations[] = {LOCATION_MZONE, LOCATION_SZONE, LOCATION_HAND, LOCATION_GRAVE,
                                              LOCATION_REMOVED, LOCATION_EXTRA};
    for (int side = 0; side < 2; ++side) {
        const int con = side == 0 ? viewer : 1 - viewer;
        std::vector<std::optional<QCard>> mzone;
        for (uint32_t loc : side_locations) {
            auto slots = parse_location(core.query_location(kCardQueryFlags, static_cast<uint8_t>(con), loc));
            for (size_t seq = 0; seq < slots.size(); ++seq) {
                const auto& c = slots[seq];
                if (!c || (loc == LOCATION_EXTRA && side == 1 && !c->is_public)) continue;
                const bool visible = c->is_public || side == 0;
                keys[{con, static_cast<int>(loc), static_cast<int64_t>(seq), -1}] = rows.size();
                rows.push_back(card_row(*c, loc, static_cast<uint32_t>(seq), side, viewer, visible, vocab));
            }
            if (loc == LOCATION_MZONE) mzone = std::move(slots);
        }
        for (size_t seq = 0; seq < mzone.size(); ++seq) {
            if (!mzone[seq]) continue;
            const auto& overlay = mzone[seq]->overlay;
            for (size_t k = 0; k < overlay.size(); ++k) {
                keys[{con, static_cast<int>(LOCATION_OVERLAY), static_cast<int64_t>(seq), static_cast<int64_t>(k)}] = rows.size();
                rows.push_back(material_row(overlay[k], static_cast<uint32_t>(seq), static_cast<uint32_t>(k), side, cards, vocab));
            }
        }
    }
    std::vector<QCard> deck;
    for (auto& c : parse_location(core.query_location(kCardQueryFlags, static_cast<uint8_t>(viewer), LOCATION_DECK)))
        if (c) deck.push_back(std::move(*c));
    std::stable_sort(deck.begin(), deck.end(),
                     [&](const QCard& a, const QCard& b) { return vocab.index(a.code) < vocab.index(b.code); });
    for (const QCard& c : deck) {
        deck_rows.emplace(c.code, rows.size());
        rows.push_back(card_row(c, LOCATION_DECK, 0, 0, viewer, true, vocab));
    }

    const size_t n_rows = std::min(rows.size(), static_cast<size_t>(N_CARDS));
    out.cards.assign(N_CARDS * F_CARD, 0);
    for (size_t i = 0; i < n_rows; ++i) std::copy(rows[i].begin(), rows[i].end(), out.cards.begin() + i * F_CARD);

    // globals
    out.globals.assign(G_GLOBAL, 0);
    auto& g = out.globals;
    const auto& lp = tracker.lp();
    g[0] = viewer;
    g[1] = viewer == 0 ? 1 : 0;
    g[2] = tracker.turn_player() == viewer ? 1 : 0;
    g[3] = clamp(tracker.turn(), 0, 999);
    g[4] = bit_index(tracker.phase());
    g[5] = clamp(lp[viewer]);
    g[6] = clamp(lp[1 - viewer]);
    static const uint32_t count_locations[] = {LOCATION_DECK, LOCATION_HAND, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA};
    for (int side = 0; side < 2; ++side) {
        const int con = side == 0 ? viewer : 1 - viewer;
        for (int i = 0; i < 5; ++i)
            g[7 + side * 5 + i] = static_cast<int32_t>(core.query_count(static_cast<uint8_t>(con), count_locations[i]));
    }
    g[17] = static_cast<int32_t>(chain_size(core.query_field()));
    g[18] = decision ? decision->type : 0;
    g[19] = tracker.state() ? static_cast<int32_t>(tracker.state()->picked_count()) : 0;
    g[20] = static_cast<int32_t>(actions.size());
    g[21] = 0;

    // actions
    out.actions.assign(MAX_OPTIONS * A_ACTION, 0);
    out.action_mask.assign(MAX_OPTIONS, 0);
    for (size_t i = 0; i < actions.size() && i < static_cast<size_t>(MAX_OPTIONS); ++i) {
        const Action& a = actions[i];
        int32_t* row = out.actions.data() + i * A_ACTION;
        out.action_mask[i] = 1;
        row[0] = static_cast<int32_t>(a.kind) + 1;
        if (a.has_card) {
            if (a.card.code) row[2] = vocab.index(a.card.code);  // 0 when hidden from the decider
            const Loc& l = a.card.loc;
            std::optional<size_t> ref;
            if (l.location & LOCATION_OVERLAY) {
                auto it = keys.find({l.controller, static_cast<int>(LOCATION_OVERLAY), l.sequence, l.position});
                if (it != keys.end()) ref = it->second;
            } else if (l.location == LOCATION_DECK) {
                if (l.controller == viewer) {
                    auto it = deck_rows.find(a.card.code);
                    if (it != deck_rows.end()) ref = it->second;
                }
            } else {
                auto it = keys.find({l.controller, l.location, l.sequence, -1});
                if (it != keys.end()) ref = it->second;
            }
            if (ref && *ref < n_rows) row[1] = static_cast<int32_t>(*ref + 1);
        }
        if (a.kind == DECLARE) row[2] = vocab.index(static_cast<uint32_t>(a.value));
        if (a.description) {
            const uint64_t code = a.description >> 20;
            if (code <= 0xFFFFFFFFull && cards.find(static_cast<uint32_t>(code))) {
                row[3] = vocab.index(static_cast<uint32_t>(code));
                row[4] = clamp(static_cast<int64_t>(a.description & 0xFFFFF) + 1);
            } else {
                row[5] = static_cast<int32_t>(std::min<uint64_t>(a.description, kClamp));
            }
        }
        if (a.kind == POSITION) {
            row[6] = position_enum(static_cast<uint32_t>(a.value));
        } else if (a.kind == PLACE) {
            const int64_t player = a.value >> 16, loc = (a.value >> 8) & 0xFF, seq = a.value & 0xFF;
            row[7] = static_cast<int32_t>((player == viewer ? 0 : 1) * 16 + (loc == LOCATION_MZONE ? 0 : 8) + seq + 1);
        }
        switch (a.kind) {
            case RACE: case ATTRIBUTE: row[8] = bit_index(static_cast<uint64_t>(a.value)); break;
            case NUMBER: case RPS: case COUNTER: case SELECT: case UNSELECT: row[8] = clamp(a.value & 0xFFFF); break;
            default: break;
        }
        row[9] = clamp(static_cast<int64_t>(a.index) + 1, 0, 255);
    }
    mask_duplicates(out.cards, out.actions, out.action_mask, decision ? decision->type : 0);
}

}  // namespace ygorl::host
