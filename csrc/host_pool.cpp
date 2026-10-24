#include "host_pool.h"

#include <algorithm>

namespace ygorl::host {

HostPool::HostPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
                   std::shared_ptr<ScriptSource> scripts, std::shared_ptr<const Vocab> vocab, bool privileged,
                   size_t event_length, bool skip_forced, bool selection_history, uint32_t cancel_budget, bool attack_history,
                   bool snapshots)
    : cards_(std::move(cards)), scripts_(std::move(scripts)), vocab_(std::move(vocab)), privileged_(privileged),
      event_length_(event_length),
      skip_forced_(skip_forced), selection_history_(selection_history), attack_history_(attack_history), cancel_budget_(cancel_budget),
      snapshots_(snapshots), slots_(num_envs), running_(new std::atomic<bool>[num_envs]),
      slot_escapes_(new std::atomic<uint64_t>[num_envs]), games_(num_envs, 0) {
    if (cancel_budget != 0 && cancel_budget != 1 && cancel_budget != 4 && cancel_budget != CANCEL_BUDGET_OFF)
        throw std::invalid_argument("cancel_budget must be 0, 1, 4 or 32");
    if (snapshots && !arena::available())
        throw std::runtime_error("snapshots need an arena build (this extension was built with -DYGORL_ARENA=OFF)");
    for (size_t i = 0; i < num_envs; ++i) running_[i] = false, slot_escapes_[i] = 0;
    pool_ = std::make_unique<WorkerPool<std::pair<int, PoolJob>, PoolEvent>>(
        num_envs, num_threads, [this](std::pair<int, PoolJob>& job) { return run(job.first, job.second); });
}

HostPool::~HostPool() {
    pool_->shutdown();  // workers stop before the slots they touch are destroyed
}

void HostPool::submit(int env, PoolJob job) { pool_->submit(env, {env, std::move(job)}); }

void HostPool::reset(int env, PoolJob job) {
    job.kind = PoolJob::RESET;
    submit(env, std::move(job));
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    ++games_[env];  // after submit: a busy env keeps its game; earlier snapshots of env no longer restore
}

void HostPool::step(int env, size_t action) {
    // Slots are only touched by their worker thread; a step on an env without a
    // running game comes back as an "error" event.
    PoolJob job;
    job.action = action;
    submit(env, std::move(job));
}

void HostPool::require_snapshots() const {
    if (!snapshots_) throw std::runtime_error("this pool was created without snapshots (pass snapshots=True)");
}

uint64_t HostPool::snapshot(int env) {
    require_snapshots();
    if (pool_->busy(env))  // also checks the range; an idle env's flags are settled (recv synchronizes with its worker)
        throw std::runtime_error("env " + std::to_string(env) + " is busy (its previous job has not been received)");
    if (!running_[env]) throw std::runtime_error("env " + std::to_string(env) + " has no running game to snapshot");
    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(snapshots_mutex_);
        id = next_snapshot_++;
        stored_[id] = Stored{env, games_[env], nullptr};
    }
    PoolJob job;
    job.kind = PoolJob::SNAPSHOT;
    job.snapshot_id = id;
    try {
        submit(env, std::move(job));
    } catch (...) {
        std::lock_guard<std::mutex> lock(snapshots_mutex_);
        stored_.erase(id);
        throw;
    }
    return id;
}

void HostPool::restore(int env, uint64_t id) {
    require_snapshots();
    if (pool_->busy(env))
        throw std::runtime_error("env " + std::to_string(env) + " is busy (its previous job has not been received)");
    PoolJob job;
    job.kind = PoolJob::RESTORE;
    {
        std::lock_guard<std::mutex> lock(snapshots_mutex_);
        auto it = stored_.find(id);
        if (it == stored_.end()) throw std::invalid_argument("unknown or released snapshot " + std::to_string(id));
        const Stored& s = it->second;
        if (s.env != env)
            throw std::invalid_argument("snapshot " + std::to_string(id) + " was taken in env " + std::to_string(s.env) +
                                        ", not env " + std::to_string(env) + " (snapshots restore only into their game)");
        if (s.game != games_[env])
            throw std::invalid_argument("snapshot " + std::to_string(id) + " was taken from an earlier game of env " +
                                        std::to_string(env) + " (snapshots restore only into their game)");
        if (!s.snapshot) throw std::invalid_argument("snapshot " + std::to_string(id) + " failed when it was taken");
        job.snapshot = s.snapshot;
    }
    submit(env, std::move(job));
}

void HostPool::release(int env, uint64_t id) {
    std::shared_ptr<const HostDuel::Snapshot> freed;  // destroyed outside the lock
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    auto it = stored_.find(id);
    if (it == stored_.end() || it->second.env != env)
        throw std::invalid_argument("env " + std::to_string(env) + " has no snapshot " + std::to_string(id));
    freed = std::move(it->second.snapshot);
    stored_.erase(it);
}

size_t HostPool::snapshot_count() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    return stored_.size();
}

size_t HostPool::snapshot_bytes() const {
    std::lock_guard<std::mutex> lock(snapshots_mutex_);
    size_t total = 0;
    for (const auto& [id, s] : stored_)
        if (s.snapshot) total += s.snapshot->nbytes();
    return total;
}

uint64_t HostPool::arena_escapes() const {
    uint64_t total = retired_escapes_.load();
    for (size_t i = 0; i < slots_.size(); ++i) total += slot_escapes_[i].load();
    return total;
}

PoolEvent HostPool::run(int env, PoolJob& job) {
    PoolEvent ev;
    ev.env_id = env;
    auto& host = slots_[env];
    auto retire = [&] {  // destroy the slot's duel, keeping its arena escape count
        if (host) retired_escapes_ += host->arena_escapes();
        slot_escapes_[env] = 0;
        host.reset();
    };
    // Without snapshots a finished game is dropped at once; with them it stays restorable until the next reset.
    auto finish = [&] {
        running_[env] = false;
        if (snapshots_)
            slot_escapes_[env] = host->arena_escapes();
        else
            retire();
    };
    std::string error;
    bool failed = false;
    try {
        switch (job.kind) {
            case PoolJob::RESET:
                running_[env] = false;
                retire();
                host = std::make_unique<HostDuel>(cards_, scripts_, vocab_, snapshots_);
                host->set_event_length(event_length_);
                host->set_selection_history(selection_history_);
                host->set_attack_history(attack_history_);
                host->set_cancel_budget(cancel_budget_);
                host->start(job.seed, job.flags, job.team1, job.team2, job.decks, job.max_turns, job.max_decisions);
                running_[env] = true;
                break;
            case PoolJob::STEP:
                if (!host || !running_[env] || host->done()) throw std::runtime_error("env has no running game");
                host->act(job.action);
                break;
            case PoolJob::SNAPSHOT: {
                // HostPool::snapshot checked the env is idle at a decision; the job still guards against misuse.
                if (!host || !running_[env] || host->done()) throw std::runtime_error("env has no running game");
                auto snap = host->snapshot();
                std::lock_guard<std::mutex> lock(snapshots_mutex_);
                auto it = stored_.find(job.snapshot_id);
                if (it != stored_.end()) it->second.snapshot = std::move(snap);  // else released before it ran
                break;
            }
            case PoolJob::RESTORE:
                if (!host) throw std::runtime_error("env has no game to restore");
                host->restore(*job.snapshot);
                running_[env] = true;
                break;
        }
        // A decision with a single choosable row has nothing to learn: take it here (same game either way). The
        // single-action case needs no encoding; otherwise the mask decides (equivalent copies are masked out).
        // Snapshots are taken at points this loop returned, so after SNAPSHOT / RESTORE it only observes.
        while (!host->done() && !host->actions().empty()) {  // actions() may stop the duel (decision limit)
            if (skip_forced_ && host->actions().size() == 1) {
                host->act(0);
                continue;
            }
            host->observe(ev.obs);
            if (!skip_forced_) break;
            const auto& mask = ev.obs.action_mask;
            if (std::count(mask.begin(), mask.end(), 1) != 1) break;
            host->act(static_cast<size_t>(std::find(mask.begin(), mask.end(), 1) - mask.begin()));
        }
        if (!host->done() && !host->actions().empty()) {
            ev.player = host->player();
            if (privileged_) {
                host->observe_privileged(ev.privileged);
                ev.has_privileged = true;
            }
            slot_escapes_[env] = host->arena_escapes();
            return ev;
        }
    } catch (const std::exception& e) {
        if (!host || !host->started()) {  // start() failed: there is no game to report on
            running_[env] = false;
            retire();
            ev.done = true;
            ev.outcome.reason = "error";
            ev.outcome.error = e.what();
            return ev;
        }
        failed = true;
        error = e.what();
    }
    ev.done = true;
    ev.outcome = host->tracker().outcome();
    if (failed) {  // a host exception ends the game as an error: a draw (engine.outcome.winner), like VecDuelEnv
        ev.outcome.reason = "error";
        ev.outcome.error = error;
        ev.outcome.winner = -1;
    }
    finish();
    return ev;
}

}  // namespace ygorl::host
