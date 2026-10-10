// Reading the core's message buffers: the [u32 length][record] framing and a bounds-checked reader of one record's
// payload. Shared by the decision decoder and the tracker (host.cpp), the event encoder (event_encoder.cpp) and the
// DuelPool's win check (duel_pool.cpp); the Python counterparts are messages.split_messages / messages.Reader.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "host.h"

namespace ygorl::host {

// A record's payload ended before a field it must have (Python decodes such a record as UndecodableMessage).
struct Truncated : std::runtime_error {
    Truncated() : std::runtime_error("truncated payload") {}
};

class MessageReader {
public:
    MessageReader(const uint8_t* data, size_t size) : p_(data), end_(data + size) {}
    template <typename T>
    T get() {
        if (static_cast<size_t>(end_ - p_) < sizeof(T)) throw Truncated();
        T v;
        std::memcpy(&v, p_, sizeof(T));
        p_ += sizeof(T);
        return v;
    }
    uint8_t u8() { return get<uint8_t>(); }
    uint16_t u16() { return get<uint16_t>(); }
    uint32_t u32() { return get<uint32_t>(); }
    int32_t i32() { return get<int32_t>(); }
    uint64_t u64() { return get<uint64_t>(); }
    void skip(size_t n) {
        if (static_cast<size_t>(end_ - p_) < n) throw Truncated();
        p_ += n;
    }
    // the three location encodings of core messages: loc_info (controller, location, u32 sequence, u32 position),
    // 32-bit sequence without a position, 8-bit sequence
    Loc loc_info() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u32();
        l.position = u32();
        return l;
    }
    Loc loc32() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u32();
        return l;
    }
    Loc loc8() {
        Loc l;
        l.controller = u8();
        l.location = u8();
        l.sequence = u8();
        return l;
    }
    CardRef card(Loc (MessageReader::*loc)()) {
        CardRef c;
        c.code = u32();
        c.loc = (this->*loc)();
        return c;
    }

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

// Calls ``f(record, length)`` for each record of a message buffer (``record[0]`` is the message type, ``length`` >= 1
// counts it). Corrupt framing (a zero or overlong length) ends the walk: the rest is one undecodable message.
template <typename F>
void for_each_record(const uint8_t* data, size_t size, F&& f) {
    size_t pos = 0;
    while (pos + 4 <= size) {
        uint32_t len;
        std::memcpy(&len, data + pos, 4);
        if (len == 0 || pos + 4 + len > size) return;
        f(data + pos + 4, static_cast<size_t>(len));
        pos += 4 + len;
    }
}

}  // namespace ygorl::host
