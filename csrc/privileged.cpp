// Training-mode ground truth: C++ mirror of ygorl/env/privileged.py (spec: docs/encoding.md).
#include "privileged.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "host.h"
#include "ocgapi_constants.h"

namespace ygorl::host {

namespace {

constexpr uint32_t kFlags = QUERY_CODE | QUERY_POSITION;  // QUERY_IS_PUBLIC is always returned

struct Slot {
    bool present = false;
    uint32_t code = 0, position = 0;
    uint8_t is_public = 0;
};

using Entry = std::array<int32_t, P_COLS>;  // card_index, public, sequence

template <typename T>
T rd(const std::string& b, size_t pos) {
    T v;
    std::memcpy(&v, b.data() + pos, sizeof(T));
    return v;
}

// Location query layout: u32 size, then per slot `u16 0` (empty) or [u16 len][u32 flag][value] chunks up to QUERY_END.
std::vector<Slot> query(Duel& core, int con, uint32_t loc) {
    const std::string buf = core.query_location(kFlags, static_cast<uint8_t>(con), loc);
    std::vector<Slot> out;
    size_t pos = 4;
    while (pos + 2 <= buf.size()) {
        Slot s;
        if (rd<uint16_t>(buf, pos) == 0) {
            out.push_back(s);
            pos += 2;
            continue;
        }
        s.present = true;
        while (pos + 6 <= buf.size()) {
            const uint16_t len = rd<uint16_t>(buf, pos);
            const uint32_t flag = rd<uint32_t>(buf, pos + 2);
            const size_t body = pos + 6;
            pos += 2 + len;
            if (flag == QUERY_END) break;
            if (flag == QUERY_CODE) s.code = rd<uint32_t>(buf, body);
            else if (flag == QUERY_POSITION) s.position = rd<uint32_t>(buf, body);
            else if (flag == QUERY_IS_PUBLIC) s.is_public = rd<uint8_t>(buf, body);
        }
        out.push_back(s);
    }
    return out;
}

Entry entry(const Slot& s, uint32_t seq, const Vocab& vocab) {
    return {vocab.index(s.code), s.is_public ? 1 : 0, static_cast<int32_t>(seq)};
}

bool facedown(const Slot& s) { return (s.position & POS_FACEDOWN) != 0; }

void fill(const std::vector<Entry>& entries, int width, std::vector<int32_t>& out) {
    out.assign(static_cast<size_t>(width) * P_COLS, 0);
    const size_t n = std::min(entries.size(), static_cast<size_t>(width));
    for (size_t i = 0; i < n; ++i) std::copy(entries[i].begin(), entries[i].end(), out.begin() + i * P_COLS);
}

}  // namespace

void encode_privileged(Duel& core, int viewer, const Vocab& vocab, Privileged& out) {
    const int op = 1 - viewer;
    std::vector<Entry> hand, deck, extra, removed;
    {
        const auto slots = query(core, op, LOCATION_HAND);
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].present) hand.push_back(entry(slots[i], static_cast<uint32_t>(i), vocab));
    }
    for (const Slot& s : query(core, op, LOCATION_DECK))
        if (s.present) deck.push_back(entry(s, 0, vocab));
    for (const Slot& s : query(core, op, LOCATION_EXTRA))
        if (s.present) extra.push_back(entry(s, 0, vocab));
    std::sort(deck.begin(), deck.end());  // lexicographic (card_index, public, 0): composition only
    std::sort(extra.begin(), extra.end());
    {
        const auto slots = query(core, op, LOCATION_REMOVED);
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].present && facedown(slots[i])) removed.push_back(entry(slots[i], static_cast<uint32_t>(i), vocab));
    }
    fill(hand, P_HAND, out.op_hand);
    fill(deck, P_DECK, out.op_deck);
    fill(extra, P_EXTRA, out.op_extra);
    fill(removed, P_REMOVED, out.op_removed);

    out.op_set.assign(static_cast<size_t>(P_SET) * P_COLS, 0);
    int32_t n_set = 0;
    const std::array<std::array<uint32_t, 3>, 2> zones{{{LOCATION_MZONE, 0, 7}, {LOCATION_SZONE, 7, 8}}};  // loc, row base, zones
    for (const auto& [loc, base, n] : zones) {
        const auto slots = query(core, op, loc);
        for (size_t seq = 0; seq < slots.size() && seq < n; ++seq) {
            if (!slots[seq].present || !facedown(slots[seq])) continue;
            const Entry e = entry(slots[seq], static_cast<uint32_t>(seq), vocab);
            std::copy(e.begin(), e.end(), out.op_set.begin() + (base + seq) * P_COLS);
            ++n_set;
        }
    }
    out.counts = {static_cast<int32_t>(hand.size()), static_cast<int32_t>(deck.size()), static_cast<int32_t>(extra.size()),
                  n_set, static_cast<int32_t>(removed.size())};
}

void HostDuel::observe_privileged(Privileged& out) {
    const int viewer = player();
    encode_privileged(*core_, viewer < 0 ? 0 : viewer, *vocab_, out);
}

}  // namespace ygorl::host
