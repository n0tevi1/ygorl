// Per-duel memory arenas for snapshot/restore (T2.8).
//
// Every allocation the rule core makes (C++ `operator new` and Lua's allocator)
// while a duel's arena is active on the current thread lands in that arena: one
// fixed slot of a single reserved address range. All allocator metadata lives
// inside the slot, so the used prefix of the slot *is* the duel's complete
// mutable state. A snapshot copies that prefix; restoring copies it back to the
// same address, so no pointer needs relocation.
//
// Rules (see docs/engine.md):
//  - Activate an arena only around calls into the core (Scope).
//  - Callbacks from the core into host code suspend it (Suspend), so host data
//    structures never point into an arena; host code that calls back into the
//    core from a callback (OCG_LoadScript) resumes it (Resume).
//  - `operator new/delete` of the whole extension are replaced (the extension
//    links libstdc++ statically, so this covers library code too); outside an
//    active arena they fall through to malloc/free. Frees are routed by address.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ygorl::arena {

// True when the extension was built with arena support (YGORL_ARENA).
bool available();

class Image;

class Arena {
public:
    // Acquire a free slot; throws std::runtime_error if arenas are unavailable or exhausted.
    static std::unique_ptr<Arena> create();
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    uint64_t id() const { return id_; }
    size_t used() const;  // bytes of the slot in use (including metadata and free blocks)
    uint64_t escapes() const { return escapes_; }
    void note_escape() { ++escapes_; }

    std::unique_ptr<Image> capture() const;
    void restore(const Image& image);  // image must come from this arena (checked by the caller)

    // Allocator entry points (used by operator new and the Lua hook).
    void* allocate(size_t size);        // nullptr when the slot is full
    static void deallocate(void* ptr);  // ptr must be owned by some arena

private:
    Arena(uint32_t slot, uint8_t* base, uint64_t id);
    uint32_t slot_;
    uint8_t* base_;
    uint64_t id_;
    uint64_t escapes_ = 0;
};

class Image {
public:
    uint64_t arena_id() const { return arena_id_; }
    size_t size() const { return size_; }

private:
    friend class Arena;
    uint64_t arena_id_ = 0;
    size_t size_ = 0;
    std::unique_ptr<uint8_t[]> data_;
};

// True if `ptr` points into the arena reservation (any slot).
bool owns(const void* ptr);

// RAII: make `arena` the active arena of this thread (null: the host heap), with nothing suspended.
class Scope {
public:
    explicit Scope(Arena* arena);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Arena* prev_active_;
    Arena* prev_suspended_;
};

// RAII: inside a core callback, route allocations to the host heap again.
class Suspend {
public:
    Suspend();
    ~Suspend();
    Suspend(const Suspend&) = delete;
    Suspend& operator=(const Suspend&) = delete;

private:
    Arena* prev_active_;
    Arena* prev_suspended_;
};

// RAII: inside a suspended callback, re-activate the suspended arena (to call into the core again).
class Resume {
public:
    Resume();
    ~Resume();
    Resume(const Resume&) = delete;
    Resume& operator=(const Resume&) = delete;

private:
    Arena* prev_active_;
    Arena* prev_suspended_;
};

}  // namespace ygorl::arena

// Lua allocator used by the patched lauxlib (patches/ygopro-core/0003-*.patch).
void* ygorl_lua_alloc(void* ud, void* ptr, size_t osize, size_t nsize);
