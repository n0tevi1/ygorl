// Generic "env i runs on thread i % T" worker pool with an async submit/recv interface.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ygorl {

template <typename Job, typename Result>
class WorkerPool {
public:
    using Handler = std::function<Result(Job&)>;

    WorkerPool(size_t num_envs, size_t num_threads, Handler handler)
        : handler_(std::move(handler)), busy_(num_envs) {
        if (num_envs == 0 || num_threads == 0) throw std::invalid_argument("num_envs and num_threads must be positive");
        for (auto& b : busy_) b = false;
        for (size_t i = 0; i < num_threads; ++i) workers_.push_back(std::make_unique<Worker>());
        for (size_t i = 0; i < num_threads; ++i) workers_[i]->thread = std::thread([this, i] { run(i); });
    }
    ~WorkerPool() { shutdown(); }

    void shutdown() {
        stopping_ = true;
        for (auto& w : workers_) {
            { std::lock_guard<std::mutex> lock(w->mutex); }
            w->cv.notify_all();
        }
        for (auto& w : workers_)
            if (w->thread.joinable()) w->thread.join();
    }

    // Marks env busy (throws if it already is) and queues the job on its thread. The env stays
    // busy until recv() hands out the job's result.
    void submit(int env, Job job) {
        check(env);
        bool expected = false;
        if (!busy_[env].compare_exchange_strong(expected, true))
            throw std::runtime_error("env " + std::to_string(env) + " is busy (its previous job has not been received)");
        Worker& w = *workers_[static_cast<size_t>(env) % workers_.size()];
        ++in_flight_;
        {
            std::lock_guard<std::mutex> lock(w.mutex);
            w.queue.emplace_back(env, std::move(job));
        }
        w.cv.notify_one();
    }

    std::vector<Result> recv(size_t min_results, int timeout_ms) {
        std::unique_lock<std::mutex> lock(results_mutex_);
        auto ready = [&] { return results_.size() >= std::min(min_results, in_flight_.load()); };
        if (timeout_ms < 0)
            results_cv_.wait(lock, ready);
        else
            results_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
        std::vector<Result> out;
        out.reserve(results_.size());
        while (!results_.empty()) {
            busy_[results_.front().first] = false;  // reusable once its result is handed out
            out.push_back(std::move(results_.front().second));
            results_.pop_front();
        }
        in_flight_ -= out.size();
        return out;
    }

    bool busy(int env) const {
        check(env);
        return busy_[env];
    }
    size_t pending() const { return in_flight_.load(); }
    size_t num_envs() const { return busy_.size(); }
    size_t num_threads() const { return workers_.size(); }

private:
    struct Worker {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::pair<int, Job>> queue;
    };

    void check(int env) const {
        if (env < 0 || static_cast<size_t>(env) >= busy_.size())
            throw std::out_of_range("env id " + std::to_string(env) + " out of range");
    }

    void run(size_t index) {
        Worker& w = *workers_[index];
        while (true) {
            std::pair<int, Job> item;
            {
                std::unique_lock<std::mutex> lock(w.mutex);
                w.cv.wait(lock, [&] { return stopping_ || !w.queue.empty(); });
                if (stopping_) return;
                item = std::move(w.queue.front());
                w.queue.pop_front();
            }
            Result result = handler_(item.second);
            {
                std::lock_guard<std::mutex> lock(results_mutex_);
                results_.emplace_back(item.first, std::move(result));
            }
            results_cv_.notify_all();
        }
    }

    Handler handler_;
    std::vector<std::atomic<bool>> busy_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::atomic<bool> stopping_{false};
    std::atomic<size_t> in_flight_{0};
    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::deque<std::pair<int, Result>> results_;
};

}  // namespace ygorl
