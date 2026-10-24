// Vectorized environment with the whole step loop in C++ (T2.2 integration, T2.7):
// each slot owns a HostDuel; a job resets it (optionally playing an action prefix, T2),
// applies one action index, or takes / restores a snapshot (T1), then returns the next
// observation (or the final result) for that env.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "host.h"
#include "privileged.h"
#include "worker_pool.h"

namespace ygorl::host {

struct PoolEvent {
    int env_id = -1;
    bool done = false;
    int player = -1;
    Observation obs;  // valid when !done
    bool has_privileged = false;
    Privileged privileged;  // training mode only (HostPool privileged = true), valid when !done
    DuelOutcome outcome;  // valid when done
    // RESET with a prefix (docs/engine.md 「前缀起局」): prefix actions applied before this event (-1: no prefix);
    // prefix_error: action prefix_applied was not legal there (outcome.error says why; the env has no game).
    int64_t prefix_applied = -1;
    bool prefix_error = false;
};

struct PoolJob {
    enum Kind { STEP, RESET, SNAPSHOT, RESTORE };
    Kind kind = STEP;
    size_t action = 0;
    std::array<uint64_t, 4> seed{};
    uint64_t flags = 0;
    PlayerOptions team1, team2;
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> decks;
    uint32_t max_turns = 0, max_decisions = 0;
    // RESET: action indices applied on the worker after the start, as successive step() calls would (same
    // skip_forced handling, no events in between); each must be a choosable row of its decision's mask.
    std::vector<uint32_t> prefix;
    uint64_t snapshot_id = 0;                          // SNAPSHOT: where to store it
    std::shared_ptr<const HostDuel::Snapshot> snapshot;  // RESTORE
};

class HostPool {
public:
    HostPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
             std::shared_ptr<ScriptSource> scripts, std::shared_ptr<const Vocab> vocab, bool privileged = false,
             size_t event_length = 0, bool skip_forced = false, bool selection_history = false, uint32_t cancel_budget = CANCEL_BUDGET_OFF, bool attack_history = false,
             bool snapshots = false);
    ~HostPool();
    void reset(int env, PoolJob job);
    void step(int env, size_t action);
    // Snapshots (pools built with snapshots = true; docs/engine.md 「步进环境快照」). Like step, snapshot and
    // restore are jobs on the env's worker: the env stays busy until recv() returns their event, which is the
    // current observation (after restore: the one the snapshot was taken at, byte for byte). snapshot() returns the
    // id at once; an env must not be busy, and must be at a decision of a running game. restore() accepts only a
    // snapshot of the env's current game (finished or not); anything else is std::invalid_argument. release() frees
    // a snapshot right away (no job; the env may be busy).
    uint64_t snapshot(int env);
    void restore(int env, uint64_t id);
    void release(int env, uint64_t id);
    // The engine responses of env's current game so far (finished games of snapshot pools included); the env must
    // be idle with a game in its slot (std::runtime_error otherwise).
    std::vector<std::string> responses(int env) const;
    size_t snapshot_count() const;
    size_t snapshot_bytes() const;  // core images of the live snapshots
    uint64_t arena_escapes() const;  // over the duels of all slots (must be 0)
    std::vector<PoolEvent> recv(size_t min_results, int timeout_ms) { return pool_->recv(min_results, timeout_ms); }
    size_t pending() const { return pool_->pending(); }

private:
    PoolEvent run(int env, PoolJob& job);
    void submit(int env, PoolJob job);
    void require_snapshots() const;

    struct Stored {
        int env = -1;
        uint64_t game = 0;                                 // games_[env] when it was taken
        std::shared_ptr<const HostDuel::Snapshot> snapshot;  // null until its job ran (or if it failed)
    };

    std::shared_ptr<CardDatabase> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::shared_ptr<const Vocab> vocab_;
    bool privileged_ = false;  // training mode: also emit opponent ground truth
    size_t event_length_ = 0;  // event tokens per observation (T2.4); 0 = none
    bool skip_forced_ = false;  // auto-play decisions with one choosable row (they never reach Python)
    bool selection_history_ = false;
    bool attack_history_ = false;
    uint32_t cancel_budget_ = CANCEL_BUDGET_OFF;
    bool snapshots_ = false;  // finished games stay in their slot (restorable) until the next reset
    std::vector<std::unique_ptr<HostDuel>> slots_;
    std::unique_ptr<std::atomic<bool>[]> running_;  // slot has a game in progress (written by its worker)
    std::unique_ptr<std::atomic<uint64_t>[]> slot_escapes_;  // arena escapes of each slot's current duel
    std::atomic<uint64_t> retired_escapes_{0};               // of the duels already destroyed
    std::vector<uint64_t> games_;                   // resets submitted per env (under snapshots_mutex_)
    mutable std::mutex snapshots_mutex_;
    std::unordered_map<uint64_t, Stored> stored_;
    uint64_t next_snapshot_ = 1;
    std::unique_ptr<WorkerPool<std::pair<int, PoolJob>, PoolEvent>> pool_;
};

}  // namespace ygorl::host
