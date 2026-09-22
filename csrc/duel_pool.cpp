#include "duel_pool.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

#include "ocgapi_constants.h"

namespace ygorl {

namespace {

constexpr uint8_t kMsgWin = MSG_WIN;

bool contains_win(const std::string& buf) {
    size_t pos = 0;
    while (pos + 5 <= buf.size()) {
        uint32_t len = 0;
        std::memcpy(&len, buf.data() + pos, sizeof(len));
        if (len == 0 || pos + 4 + len > buf.size()) return false;
        if (static_cast<uint8_t>(buf[pos + 4]) == kMsgWin) return true;
        pos += 4 + len;
    }
    return false;
}

}  // namespace

DuelPool::DuelPool(size_t num_envs, size_t num_threads, std::shared_ptr<CardSource> cards,
                   std::shared_ptr<ScriptSource> scripts)
    : cards_(std::move(cards)), scripts_(std::move(scripts)) {
    if (num_envs == 0 || num_threads == 0) throw std::invalid_argument("num_envs and num_threads must be positive");
    if (!cards_ || !scripts_) throw std::invalid_argument("card and script sources are required");
    slots_.reserve(num_envs);
    for (size_t i = 0; i < num_envs; ++i) slots_.push_back(std::make_unique<Slot>());
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) workers_.push_back(std::make_unique<Worker>());
    for (size_t i = 0; i < num_threads; ++i) workers_[i]->thread = std::thread([this, i] { run_worker(i); });
}

DuelPool::~DuelPool() {
    stopping_ = true;
    for (auto& w : workers_) {
        { std::lock_guard<std::mutex> lock(w->mutex); }
        w->cv.notify_all();
    }
    for (auto& w : workers_)
        if (w->thread.joinable()) w->thread.join();
    // Slots (and their duels) are destroyed after every worker has stopped.
}

DuelPool::Slot& DuelPool::slot_for_submit(int env) {
    if (env < 0 || static_cast<size_t>(env) >= slots_.size())
        throw std::out_of_range("env id " + std::to_string(env) + " out of range");
    Slot& slot = *slots_[env];
    bool expected = false;
    if (!slot.busy.compare_exchange_strong(expected, true))
        throw std::runtime_error("env " + std::to_string(env) + " is busy (its previous job has not been received)");
    return slot;
}

void DuelPool::submit(Job job) {
    Worker& w = *workers_[static_cast<size_t>(job.env) % workers_.size()];
    ++in_flight_;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        w.queue.push_back(std::move(job));
    }
    w.cv.notify_one();
}

void DuelPool::start(int env, const std::array<uint64_t, 4>& seed, uint64_t flags, const PlayerOptions& team1,
                     const PlayerOptions& team2, DeckLists decks) {
    if (decks.size() != 2) throw std::invalid_argument("decks must hold two (main, extra) pairs");
    slot_for_submit(env);
    Job job;
    job.env = env;
    job.is_start = true;
    job.spec = StartSpec{seed, flags, team1, team2, std::move(decks)};
    submit(std::move(job));
}

void DuelPool::respond(int env, std::string response) {
    Slot& slot = slot_for_submit(env);
    if (!slot.duel) {
        slot.busy = false;
        throw std::runtime_error("env " + std::to_string(env) + " not started");
    }
    Job job;
    job.env = env;
    job.response = std::move(response);
    submit(std::move(job));
}

void DuelPool::close_env(int env) {
    Slot& slot = slot_for_submit(env);
    slot.duel.reset();
    slot.busy = false;
}

void DuelPool::advance(Duel& duel, PoolResult& out) {
    while (true) {
        int status = duel.process();
        std::string buf = duel.get_message();
        for (auto& entry : duel.pop_logs()) out.logs.push_back(std::move(entry));
        bool win = contains_win(buf);
        out.buffer += buf;
        out.status = status;
        if (win || status != OCG_DUEL_STATUS_CONTINUE) return;
    }
}

PoolResult DuelPool::execute(Job& job) {
    PoolResult out;
    out.env_id = job.env;
    Slot& slot = *slots_[job.env];
    try {
        if (job.is_start) {
            auto& s = job.spec;
            slot.duel.reset();
            auto duel = std::make_unique<Duel>(s.seed, s.flags, s.team1, s.team2, cards_, scripts_);
            for (const char* base : {"constant.lua", "utility.lua"})
                if (!duel->load_script(base)) throw std::runtime_error(std::string("failed to load base script ") + base);
            for (uint8_t team = 0; team < 2; ++team) {
                for (uint32_t code : s.decks[team].first)
                    duel->new_card(team, 0, code, team, LOCATION_DECK, 0, POS_FACEDOWN_DEFENSE);
                for (uint32_t code : s.decks[team].second)
                    duel->new_card(team, 0, code, team, LOCATION_EXTRA, 0, POS_FACEDOWN_DEFENSE);
            }
            duel->start();
            slot.duel = std::move(duel);
        } else {
            slot.duel->set_response(job.response);
        }
        advance(*slot.duel, out);
    } catch (const std::exception& e) {
        out.status = -1;
        out.error = e.what();
    } catch (...) {
        out.status = -1;
        out.error = "unknown error";
    }
    return out;
}

void DuelPool::run_worker(size_t index) {
    Worker& w = *workers_[index];
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(w.mutex);
            w.cv.wait(lock, [&] { return stopping_ || !w.queue.empty(); });
            if (stopping_) return;
            job = std::move(w.queue.front());
            w.queue.pop_front();
        }
        PoolResult result = execute(job);
        slots_[job.env]->busy = false;  // before publishing: after recv() the slot is reusable
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(std::move(result));
        }
        results_cv_.notify_all();
    }
}

std::vector<PoolResult> DuelPool::recv(size_t min_results, int timeout_ms) {
    std::unique_lock<std::mutex> lock(results_mutex_);
    // in_flight_ counts jobs not yet received: never wait for more than can still arrive.
    auto ready = [&] { return results_.size() >= std::min(min_results, in_flight_.load()); };
    if (timeout_ms < 0)
        results_cv_.wait(lock, ready);
    else
        results_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
    std::vector<PoolResult> out;
    out.reserve(results_.size());
    while (!results_.empty()) {
        out.push_back(std::move(results_.front()));
        results_.pop_front();
    }
    in_flight_ -= out.size();
    return out;
}

}  // namespace ygorl
