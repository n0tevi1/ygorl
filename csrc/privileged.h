// Training-mode ground truth about the opponent (T2.5): C++ mirror of
// ygorl/env/privileged.py (spec: docs/encoding.md, "训练态真值").
//
// These tensors are for the critic and belief-head losses only; they never go
// into the actor's Observation. HostPool computes them only when constructed
// with privileged = true.
#pragma once

#include <cstdint>
#include <vector>

#include "core_backend.h"

namespace ygorl::host {

class Vocab;

constexpr int P_HAND = 32, P_DECK = 64, P_EXTRA = 32, P_SET = 15, P_REMOVED = 64, P_COLS = 3, P_COUNTS = 5;
constexpr int P_NEXT = 10;  // deck order: the next draws of each player, top first

// Each list is [width, P_COLS] row-major with rows (card_index, public, sequence).
struct Privileged {
    std::vector<int32_t> op_hand, op_deck, op_extra, op_set, op_removed, counts, my_next, op_next;
};

// Opponent (1 - viewer) hand, main deck, extra deck, face-down field cards and face-down banished cards; the next
// P_NEXT draws of both players (my_next / op_next).
void encode_privileged(Duel& core, int viewer, const Vocab& vocab, Privileged& out);

}  // namespace ygorl::host
