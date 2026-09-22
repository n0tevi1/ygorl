// Vectorized environment with the whole step loop in C++ (T2.2 integration, T2.7):
// each slot owns a HostDuel; a job resets it or applies one action index, then
// returns the next observation (or the final result) for that env.
#pragma once

#include <memory>

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
    // final result (valid when done)
    int winner = -1, win_reason = -1;
    std::string reason, error;
    uint32_t turns = 0, decisions = 0;
    std::array<int64_t, 2> lp{};
    std::vector<std::string> responses;
};

struct PoolJob {
    bool reset = false;
    size_t action = 0;
    std::array<uint64_t, 4> seed{};
    uint64_t flags = 0;
    PlayerOptions team1, team2;
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> decks;
    uint32_t max_turns = 0, max_decisions = 0;
};

class HostPool {
public:
    HostPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
             std::shared_ptr<ScriptSource> scripts, std::shared_ptr<const Vocab> vocab, bool privileged = false);
    ~HostPool();
    void reset(int env, PoolJob job);
    void step(int env, size_t action);
    std::vector<PoolEvent> recv(size_t min_results, int timeout_ms) { return pool_->recv(min_results, timeout_ms); }
    size_t pending() const { return pool_->pending(); }

private:
    PoolEvent run(int env, PoolJob& job);

    std::shared_ptr<CardDatabase> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::shared_ptr<const Vocab> vocab_;
    bool privileged_ = false;  // training mode: also emit opponent ground truth
    std::vector<std::unique_ptr<HostDuel>> slots_;
    std::unique_ptr<WorkerPool<std::pair<int, PoolJob>, PoolEvent>> pool_;
};

}  // namespace ygorl::host
