#include "host_pool.h"

#include <algorithm>

namespace ygorl::host {

HostPool::HostPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardDatabase> cards,
                   std::shared_ptr<ScriptSource> scripts, std::shared_ptr<const Vocab> vocab, bool privileged,
                   size_t event_length, bool skip_forced)
    : cards_(std::move(cards)), scripts_(std::move(scripts)), vocab_(std::move(vocab)), privileged_(privileged),
      event_length_(event_length),
      skip_forced_(skip_forced),
      slots_(num_envs) {
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
    try {
        if (job.reset) {
            host = std::make_unique<HostDuel>(cards_, scripts_, vocab_);
            host->set_event_length(event_length_);
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
            ev.reason = "error";
            ev.error = e.what();
            return ev;
        }
        ev.error = e.what();
    }
    const Tracker& t = host->tracker();
    ev.done = true;
    ev.winner = t.winner();
    ev.win_reason = t.win_reason();
    ev.reason = ev.error.empty() ? t.reason() : "error";
    if (ev.error.empty()) ev.error = t.error();
    ev.turns = t.turn();
    ev.decisions = t.decisions();
    ev.lp = t.lp();
    ev.responses = t.responses();
    ev.retries = t.retries();
    ev.unknown_messages = t.unknown_messages();
    ev.script_errors = t.script_errors();
    host.reset();
    return ev;
}

}  // namespace ygorl::host
