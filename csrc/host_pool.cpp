#include "host_pool.h"

#include <algorithm>

namespace ygorl::host {

HostPool::HostPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
                   std::shared_ptr<ScriptSource> scripts, std::shared_ptr<const Vocab> vocab, bool privileged,
                   size_t event_length, bool skip_forced, bool selection_history, uint32_t cancel_budget)
    : cards_(std::move(cards)), scripts_(std::move(scripts)), vocab_(std::move(vocab)), privileged_(privileged),
      event_length_(event_length),
      skip_forced_(skip_forced), selection_history_(selection_history), cancel_budget_(cancel_budget),
      slots_(num_envs) {
    if (cancel_budget != 0 && cancel_budget != 1 && cancel_budget != 4 && cancel_budget != CANCEL_BUDGET_OFF)
        throw std::invalid_argument("cancel_budget must be 0, 1, 4 or 32");
    pool_ = std::make_unique<WorkerPool<std::pair<int, PoolJob>, PoolEvent>>(
        num_envs, num_threads, [this](std::pair<int, PoolJob>& job) { return run(job.first, job.second); });
}

HostPool::~HostPool() {
    pool_->shutdown();  // workers stop before the slots they touch are destroyed
}

void HostPool::reset(int env, PoolJob job) {
    job.reset = true;
    pool_->submit(env, {env, std::move(job)});
}

void HostPool::step(int env, size_t action) {
    // Slots are only touched by their worker thread; a step on an env without a
    // running game comes back as an "error" event.
    PoolJob job;
    job.action = action;
    pool_->submit(env, {env, std::move(job)});
}

PoolEvent HostPool::run(int env, PoolJob& job) {
    PoolEvent ev;
    ev.env_id = env;
    auto& host = slots_[env];
    std::string error;
    try {
        if (job.reset) {
            host = std::make_unique<HostDuel>(cards_, scripts_, vocab_);
            host->set_event_length(event_length_);
            host->set_selection_history(selection_history_);
            host->set_cancel_budget(cancel_budget_);
            host->start(job.seed, job.flags, job.team1, job.team2, job.decks, job.max_turns, job.max_decisions);
        } else {
            if (!host || host->done()) throw std::runtime_error("env has no running game");
            host->act(job.action);
        }
        // A decision with a single choosable row has nothing to learn: take it here (same game either way). The
        // single-action case needs no encoding; otherwise the mask decides (equivalent copies are masked out).
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
            return ev;
        }
    } catch (const std::exception& e) {
        if (!host || !host->started()) {  // start() failed: there is no game to report on
            host.reset();
            ev.done = true;
            ev.outcome.reason = "error";
            ev.outcome.error = e.what();
            return ev;
        }
        error = e.what();
    }
    ev.done = true;
    ev.outcome = host->tracker().outcome();
    if (!error.empty()) {  // a host exception ends the game as an error (the tracker's winner of the moment stays)
        ev.outcome.reason = "error";
        ev.outcome.error = error;
    }
    host.reset();
    return ev;
}

}  // namespace ygorl::host
