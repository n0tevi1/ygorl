// Per-duel memory arenas (see arena.h). Deliberately small: a size-class free
// list allocator whose whole state lives at the start of its slot.
#include "arena.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ygorl::arena {
namespace {

constexpr size_t kSlotBytes = size_t(1) << 28;  // 256 MiB of address space per duel (committed lazily)
constexpr size_t kMaxSlots = size_t(1) << 14;   // up to 4 TiB reserved as PROT_NONE
constexpr size_t kMinSlots = 64;
constexpr uint64_t kSlotMagic = 0x594752'4C415245ull;  // "YGRLARE"
constexpr uint32_t kUsedMagic = 0xA7E4A7E4u;
constexpr uint32_t kFreeMagic = 0xF4EEF4EEu;
constexpr int kSmallClasses = 64;  // 16 .. 1024 bytes in 16-byte steps
constexpr int kClasses = kSmallClasses + 18;  // then powers of two: 2 KiB .. 256 MiB

struct SlotHeader {
    uint64_t magic;
    uint8_t* top;
    uint8_t* limit;
    void* free_heads[kClasses];
};

struct BlockHeader {  // 16 bytes, keeps payloads 16-byte aligned
    uint32_t cls;
    uint32_t magic;
    uint64_t reserved;
};
static_assert(sizeof(BlockHeader) == 16, "block header must keep 16-byte alignment");

constexpr size_t kHeapOffset = (sizeof(SlotHeader) + 63) & ~size_t(63);

std::atomic<uint8_t*> g_base{nullptr};
std::atomic<size_t> g_slot_count{0};
std::mutex g_mutex;
std::vector<uint32_t>* g_free_slots = nullptr;  // allocated on the host heap under g_mutex
uint32_t g_next_slot = 0;
std::atomic<uint64_t> g_next_id{1};

thread_local Arena* t_active = nullptr;
thread_local Arena* t_suspended = nullptr;

int class_of(size_t total) {
    if (total <= 1024) return static_cast<int>((total + 15) / 16) - 1;
    int bits = 64 - __builtin_clzll(static_cast<unsigned long long>(total - 1));  // ceil(log2(total))
    return kSmallClasses + (bits - 11);
}

size_t class_size(int cls) {
    return cls < kSmallClasses ? size_t(cls + 1) * 16 : size_t(1) << (cls - kSmallClasses + 11);
}

[[noreturn]] void fatal(const char* what) {
    std::fprintf(stderr, "ygorl arena: %s\n", what);
    std::abort();
}

SlotHeader* slot_of(const void* ptr) {
    auto* base = g_base.load(std::memory_order_relaxed);
    size_t index = static_cast<size_t>(static_cast<const uint8_t*>(ptr) - base) / kSlotBytes;
    return reinterpret_cast<SlotHeader*>(base + index * kSlotBytes);
}

void* slot_allocate(SlotHeader* h, size_t size) {
    size_t total = size + sizeof(BlockHeader);
    if (total < size) return nullptr;
    int cls = class_of(total);
    if (cls >= kClasses) return nullptr;
    auto* block = static_cast<uint8_t*>(h->free_heads[cls]);
    if (block) {
        h->free_heads[cls] = *reinterpret_cast<void**>(block + sizeof(BlockHeader));
    } else {
        size_t bytes = class_size(cls);
        if (static_cast<size_t>(h->limit - h->top) < bytes) return nullptr;
        block = h->top;
        h->top += bytes;
    }
    auto* bh = reinterpret_cast<BlockHeader*>(block);
    bh->cls = static_cast<uint32_t>(cls);
    bh->magic = kUsedMagic;
    return block + sizeof(BlockHeader);
}

BlockHeader* block_of(void* ptr) {
    auto* bh = reinterpret_cast<BlockHeader*>(static_cast<uint8_t*>(ptr) - sizeof(BlockHeader));
    if (bh->magic != kUsedMagic) fatal(bh->magic == kFreeMagic ? "double free" : "corrupt block header");
    return bh;
}

void* map_slot(uint32_t slot, int prot) {
    auto* addr = g_base.load() + size_t(slot) * kSlotBytes;
    void* p = mmap(addr, kSlotBytes, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}

// Host-heap only: used for arena bookkeeping, never while an arena is active.
struct HostHeap {
    HostHeap() : a_(t_active), s_(t_suspended) { t_active = t_suspended = nullptr; }
    ~HostHeap() { t_active = a_, t_suspended = s_; }
    Arena *a_, *s_;
};

void* reallocate(void* ptr, size_t osize, size_t nsize) {
    Arena* active = t_active;
    bool in_arena = ptr && owns(ptr);
    if (!active && !in_arena) return std::realloc(ptr, nsize);
    if (in_arena) {
        BlockHeader* bh = block_of(ptr);
        size_t total = nsize + sizeof(BlockHeader);
        if (total >= nsize && class_of(total) == static_cast<int>(bh->cls)) return ptr;
        void* fresh = active ? active->allocate(nsize) : slot_allocate(slot_of(ptr), nsize);
        if (!fresh) return nullptr;
        std::memcpy(fresh, ptr, std::min(osize, nsize));
        Arena::deallocate(ptr);
        return fresh;
    }
    void* fresh = active->allocate(nsize);
    if (!fresh) return nullptr;
    if (ptr) {  // a host-heap block grown inside an arena: its old contents were outside every snapshot
        active->note_escape();
        std::memcpy(fresh, ptr, std::min(osize, nsize));
        std::free(ptr);
    }
    return fresh;
}

}  // namespace

bool available() {
#ifdef YGORL_ARENA
    return true;
#else
    return false;
#endif
}

bool owns(const void* ptr) {
    auto* base = g_base.load(std::memory_order_relaxed);
    if (!base || !ptr) return false;
    auto offset = static_cast<size_t>(static_cast<const uint8_t*>(ptr) - base);
    return static_cast<const uint8_t*>(ptr) >= base &&
           offset < g_slot_count.load(std::memory_order_relaxed) * kSlotBytes;
}

std::unique_ptr<Arena> Arena::create() {
    if (!available()) throw std::runtime_error("this build has no arena support (snapshots need YGORL_ARENA)");
    HostHeap host;
    uint32_t slot;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_base.load()) {
            for (size_t slots = kMaxSlots; slots >= kMinSlots; slots /= 2) {
                void* p = mmap(nullptr, slots * kSlotBytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                               -1, 0);
                if (p != MAP_FAILED) {
                    g_slot_count.store(slots);
                    g_base.store(static_cast<uint8_t*>(p));
                    break;
                }
            }
            if (!g_base.load()) throw std::runtime_error("could not reserve address space for duel arenas");
            g_free_slots = new std::vector<uint32_t>();
        }
        if (!g_free_slots->empty()) {
            slot = g_free_slots->back();
            g_free_slots->pop_back();
        } else if (g_next_slot < g_slot_count.load()) {
            slot = g_next_slot++;
        } else {
            throw std::runtime_error("all " + std::to_string(g_slot_count.load()) + " duel arenas are in use");
        }
    }
    auto* base = static_cast<uint8_t*>(map_slot(slot, PROT_READ | PROT_WRITE));
    if (!base) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_free_slots->push_back(slot);
        throw std::runtime_error("could not map a duel arena");
    }
    auto* h = reinterpret_cast<SlotHeader*>(base);
    std::memset(h, 0, sizeof(SlotHeader));
    h->magic = kSlotMagic;
    h->top = base + kHeapOffset;
    h->limit = base + kSlotBytes;
    return std::unique_ptr<Arena>(new Arena(slot, base, g_next_id.fetch_add(1)));
}

Arena::Arena(uint32_t slot, uint8_t* base, uint64_t id) : slot_(slot), base_(base), id_(id) {}

Arena::~Arena() {
    HostHeap host;
    map_slot(slot_, PROT_NONE);  // drops every page of the slot
    std::lock_guard<std::mutex> lock(g_mutex);
    g_free_slots->push_back(slot_);
}

size_t Arena::used() const {
    return static_cast<size_t>(reinterpret_cast<const SlotHeader*>(base_)->top - base_);
}

void* Arena::allocate(size_t size) { return slot_allocate(reinterpret_cast<SlotHeader*>(base_), size); }

void Arena::deallocate(void* ptr) {
    BlockHeader* bh = block_of(ptr);
    SlotHeader* h = slot_of(ptr);
    bh->magic = kFreeMagic;
    auto* block = reinterpret_cast<uint8_t*>(bh);
    *reinterpret_cast<void**>(block + sizeof(BlockHeader)) = h->free_heads[bh->cls];
    h->free_heads[bh->cls] = block;
}

std::unique_ptr<Image> Arena::capture() const {
    HostHeap host;
    auto image = std::make_unique<Image>();
    image->arena_id_ = id_;
    image->size_ = used();
    image->data_.reset(new uint8_t[image->size_]);
    std::memcpy(image->data_.get(), base_, image->size_);
    return image;
}

void Arena::restore(const Image& image) {
    if (image.arena_id_ != id_) throw std::invalid_argument("image belongs to another arena");
    uint8_t* old_top = reinterpret_cast<SlotHeader*>(base_)->top;
    std::memcpy(base_, image.data_.get(), image.size_);
    uint8_t* new_top = reinterpret_cast<SlotHeader*>(base_)->top;
    constexpr uintptr_t kPage = 4096;
    auto from = (reinterpret_cast<uintptr_t>(new_top) + kPage - 1) & ~(kPage - 1);
    auto to = reinterpret_cast<uintptr_t>(old_top) & ~(kPage - 1);
    if (to > from + (uintptr_t(1) << 20))  // give back more than 1 MiB of pages the restored state never touched
        madvise(reinterpret_cast<void*>(from), to - from, MADV_DONTNEED);
}

Scope::Scope(Arena* arena) : prev_active_(t_active), prev_suspended_(t_suspended) {
    // Also for arena == nullptr: a plain duel called from another duel's callback must neither
    // allocate in, nor later Resume, the caller's (suspended) arena.
    t_active = arena;
    t_suspended = nullptr;
}
Scope::~Scope() {
    t_active = prev_active_;
    t_suspended = prev_suspended_;
}

Suspend::Suspend() : prev_active_(t_active), prev_suspended_(t_suspended) {
    if (t_active) {
        t_suspended = t_active;
        t_active = nullptr;
    }
}
Suspend::~Suspend() {
    t_active = prev_active_;
    t_suspended = prev_suspended_;
}

Resume::Resume() : prev_active_(t_active), prev_suspended_(t_suspended) {
    if (!t_active && t_suspended) {
        t_active = t_suspended;
        t_suspended = nullptr;
    }
}
Resume::~Resume() {
    t_active = prev_active_;
    t_suspended = prev_suspended_;
}

}  // namespace ygorl::arena

void* ygorl_lua_alloc(void* /*ud*/, void* ptr, size_t osize, size_t nsize) {
    using namespace ygorl::arena;
    if (nsize == 0) {
        if (ptr) {
            if (owns(ptr))
                Arena::deallocate(ptr);
            else
                std::free(ptr);
        }
        return nullptr;
    }
    return reallocate(ptr, ptr ? osize : 0, nsize);
}

// ------------------------------------------------------------ operator new/delete
#ifdef YGORL_ARENA
namespace {

void* allocate_or_null(size_t size) noexcept {
    if (ygorl::arena::Arena* active = ygorl::arena::t_active) return active->allocate(size);
    return std::malloc(size ? size : 1);
}

void* allocate_or_throw(size_t size) {
    void* p = allocate_or_null(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* allocate_aligned(size_t size, std::align_val_t align) noexcept {
    auto a = static_cast<size_t>(align);
    if (a <= 16) return allocate_or_null(size);
    if (ygorl::arena::Arena* active = ygorl::arena::t_active) active->note_escape();
    void* p = nullptr;
    return posix_memalign(&p, a, size ? size : 1) == 0 ? p : nullptr;
}

void release(void* ptr) noexcept {
    if (!ptr) return;
    if (ygorl::arena::owns(ptr))
        ygorl::arena::Arena::deallocate(ptr);
    else
        std::free(ptr);
}

}  // namespace

void* operator new(std::size_t size) { return allocate_or_throw(size); }
void* operator new[](std::size_t size) { return allocate_or_throw(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return allocate_or_null(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return allocate_or_null(size); }
void* operator new(std::size_t size, std::align_val_t align) {
    void* p = allocate_aligned(size, align);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t size, std::align_val_t align) { return ::operator new(size, align); }
void* operator new(std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
    return allocate_aligned(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align, const std::nothrow_t&) noexcept {
    return allocate_aligned(size, align);
}

void operator delete(void* ptr) noexcept { release(ptr); }
void operator delete[](void* ptr) noexcept { release(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { release(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { release(ptr); }
void operator delete(void* ptr, const std::nothrow_t&) noexcept { release(ptr); }
void operator delete[](void* ptr, const std::nothrow_t&) noexcept { release(ptr); }
void operator delete(void* ptr, std::align_val_t) noexcept { release(ptr); }
void operator delete[](void* ptr, std::align_val_t) noexcept { release(ptr); }
void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept { release(ptr); }
void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept { release(ptr); }
void operator delete(void* ptr, std::align_val_t, const std::nothrow_t&) noexcept { release(ptr); }
void operator delete[](void* ptr, std::align_val_t, const std::nothrow_t&) noexcept { release(ptr); }
#endif
