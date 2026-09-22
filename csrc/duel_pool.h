// Thread pool that advances many duels in parallel (T2.1).
//
// Each environment slot owns one core handle and is pinned to worker
// `env % num_threads`, so a duel is only ever touched by one thread (and a
// per-thread memory arena can be added later for snapshots, T2.8). Work is
// submitted asynchronously (start/respond) and collected with recv(), in the
// style of envpool. A job runs the core until it needs a response, ends, or
// emits MSG_WIN (the host stops at the first win, like EDOPro), and returns
// the concatenated message buffers.
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core_backend.h"

namespace ygorl {

using DeckLists = std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>;  // per team: (main, extra)

struct PoolResult {
    int env_id = -1;
    int status = -1;  // OCG_DUEL_STATUS_* of the last process(), -1 on error
    std::string buffer;
    std::vector<LogEntry> logs;
    std::string error;
};

class DuelPool {
public:
    DuelPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardSource> cards,
             std::shared_ptr<ScriptSource> scripts);
    ~DuelPool();
    DuelPool(const DuelPool&) = delete;
    DuelPool& operator=(const DuelPool&) = delete;

    // Create the duel for `env` (base scripts, cards in the given order, start) and run it
    // to its first stop. The slot must be idle; an existing duel in it is replaced.
    void start(int env, const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
               const PlayerOptions& team2, DeckLists decks);
    // Send a response to the pending decision of `env` and run to the next stop.
    void respond(int env, std::string response);
    // Wait until at least `min_results` results are ready (or `timeout_ms` passes; < 0 waits
    // forever, but never longer than it takes for all in-flight jobs to finish).
    std::vector<PoolResult> recv(size_t min_results, int timeout_ms);
    // Destroy the duel of an idle slot.
    void close_env(int env);
    size_t pending() const { return in_flight_.load(); }
    size_t num_envs() const { return slots_.size(); }
    size_t num_threads() const { return workers_.size(); }

private:
    struct StartSpec {
        std::array<uint64_t, 4> seed{};
        uint64_t flags = 0;
        PlayerOptions team1, team2;
        DeckLists decks;
    };
    struct Job {
        int env = -1;
        bool is_start = false;
        StartSpec spec;
        std::string response;
    };
    struct Slot {
        std::unique_ptr<Duel> duel;
        std::atomic<bool> busy{false};
    };
    struct Worker {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<Job> queue;
    };

    Slot& slot_for_submit(int env);
    void submit(Job job);
    void run_worker(size_t index);
    PoolResult execute(Job& job);
    void advance(Duel& duel, PoolResult& out);

    std::shared_ptr<CardSource> cards_;
    std::shared_ptr<ScriptSource> scripts_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<size_t> in_flight_{0};
    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::deque<PoolResult> results_;
};

}  // namespace ygorl
