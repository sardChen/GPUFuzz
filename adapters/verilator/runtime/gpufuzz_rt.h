// GPUFuzz Verilator runtime: signal reflection, memory models, trace writer.
//
// This header is target independent.  The generated simulator main
// (adapters/verilator/gen.py) includes it and instantiates:
//
//   * SignalTable  -- resolves RTL signals by hierarchical path and samples them
//   * StateTable   -- samples state elements (cache lines, slots, ...) per object
//   * MemModel     -- the memory the GPU sees, driven through AXI4/AXI-Lite/simple
//   * Injector     -- execution-relative event injection (backpressure, flush, ...)
//   * TraceWriter  -- JSONL trace in the format consumed by adapters/trace
//
// Signal access uses Verilator's scope symbol table, so the generated code does
// not depend on mangled internal names.  Requirements:
//   * Verilator >= 5.0
//   * --trace (or --trace-fst) so that signals appear in the symbol table
//   * the top module name and hierarchical paths from the target's signal map
#ifndef GPUFUZZ_RT_H
#define GPUFUZZ_RT_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "verilated.h"
#include "verilated_scope.h"

namespace gpufuzz {

// --------------------------------------------------------------------------
// signal access by hierarchical path
// --------------------------------------------------------------------------
struct SignalRef {
    std::string path;
    VerilatedVar* var = nullptr;
    VerilatedScope* scope = nullptr;
    bool resolved() const { return var != nullptr; }
};

static inline VerilatedVar* resolve_var(VerilatedModel* model, const std::string& path,
                                        VerilatedScope** out_scope = nullptr) {
    const size_t pos = path.rfind('.');
    const std::string scope_name = (pos == std::string::npos) ? "" : path.substr(0, pos);
    const std::string leaf = (pos == std::string::npos) ? path : path.substr(pos + 1);
    VerilatedScope* scope = nullptr;
#if defined(GPUFUZZ_SCOPE_API_CONTEXT)
    scope = Verilated::threadContextp()->scopeFind(scope_name);
#else
    scope = model->scopeFind(scope_name);
#endif
    if (!scope) return nullptr;
    VerilatedVar* var = scope->scopeFindVar(leaf.c_str());
    if (out_scope) *out_scope = scope;
    return var;
}

static inline uint64_t read_var(const VerilatedVar* var) {
    if (!var) return 0;
    const void* dp = var->datap();
    if (!dp) return 0;
    switch (var->dtype()) {
        case VLVT_UINT8:  return *static_cast<const uint8_t*>(dp);
        case VLVT_UINT16: return *static_cast<const uint16_t*>(dp);
        case VLVT_UINT32: return *static_cast<const uint32_t*>(dp);
        case VLVT_UINT64: return *static_cast<const uint64_t*>(dp);
        case VLVT_WDATA: {
            // wide value: little endian limbs, we only need the low 64 bits
            const EData* words = static_cast<const EData*>(dp);
            uint64_t v = 0;
            for (int i = 3; i >= 0; --i) v = (v << 16) | (words[i] & 0xffff);
            return v;
        }
        default: return 0;
    }
}

static inline void write_var(VerilatedVar* var, uint64_t value) {
    if (!var) return;
    void* dp = var->datap();
    if (!dp) return;
    switch (var->dtype()) {
        case VLVT_UINT8:  *static_cast<uint8_t*>(dp)  = (uint8_t)value;  break;
        case VLVT_UINT16: *static_cast<uint16_t*>(dp) = (uint16_t)value; break;
        case VLVT_UINT32: *static_cast<uint32_t*>(dp) = (uint32_t)value; break;
        case VLVT_UINT64: *static_cast<uint64_t*>(dp) = value;           break;
        default: break;
    }
}

// --------------------------------------------------------------------------
// trace writer (JSONL, one record per line)
// --------------------------------------------------------------------------
class TraceWriter {
  public:
    bool open(const std::string& path) {
        fh_ = std::fopen(path.c_str(), "w");
        return fh_ != nullptr;
    }
    ~TraceWriter() { if (fh_) std::fclose(fh_); }

    void header(const std::string& target, const std::string& top,
                const std::string& test, const std::vector<std::string>& unresolved) {
        std::fprintf(fh_, "{\"t\":\"hdr\",\"target\":\"%s\",\"top\":\"%s\",\"test\":\"%s\"",
                     target.c_str(), top.c_str(), test.c_str());
        std::fprintf(fh_, ",\"unresolved\":[");
        for (size_t i = 0; i < unresolved.size(); ++i)
            std::fprintf(fh_, "%s\"%s\"", i ? "," : "", unresolved[i].c_str());
        std::fprintf(fh_, "]}\n");
    }

    // delta encoded signal samples: only signals whose value changed
    void signals(uint64_t cycle, const std::vector<std::pair<int, uint64_t>>& changed) {
        if (changed.empty()) return;
        std::fprintf(fh_, "{\"t\":\"sig\",\"c\":%llu,\"v\":{", (unsigned long long)cycle);
        for (size_t i = 0; i < changed.size(); ++i)
            std::fprintf(fh_, "%s\"%d\":%llu", i ? "," : "", changed[i].first,
                         (unsigned long long)changed[i].second);
        std::fprintf(fh_, "}}\n");
    }

    void state(uint64_t cycle, const char* sid, const char* obj, uint64_t value,
               const char* sig) {
        std::fprintf(fh_, "{\"t\":\"st\",\"c\":%llu,\"s\":\"%s\",\"o\":\"%s\",\"v\":%llu,"
                          "\"sig\":\"%s\"}\n",
                     (unsigned long long)cycle, sid, obj, (unsigned long long)value, sig);
    }

    void event(uint64_t cycle, const char* type, const char* obj, const char* outcome) {
        std::fprintf(fh_, "{\"t\":\"ev\",\"c\":%llu,\"a\":\"%s\",\"u\":\"%s\",\"o\":\"%s\"}\n",
                     (unsigned long long)cycle, type, obj, outcome);
    }

    void arch_begin() { std::fprintf(fh_, "{\"t\":\"arch\",\"mem\":{"); }
    void arch_mem(uint64_t addr, uint32_t value, bool first) {
        std::fprintf(fh_, "%s\"%llu\":%u", first ? "" : ",", (unsigned long long)addr, value);
    }
    void arch_end(uint64_t cycles, bool timeout, int exit_code) {
        std::fprintf(fh_, "},\"regs\":{},\"status\":{\"exit\":%d},\"c\":%llu}\n",
                     exit_code, (unsigned long long)cycles);
        std::fprintf(fh_, "{\"t\":\"end\",\"c\":%llu,\"timeout\":%d,\"exit\":%d}\n",
                     (unsigned long long)cycles, timeout ? 1 : 0, exit_code);
        std::fflush(fh_);
    }

  private:
    FILE* fh_ = nullptr;
};

// --------------------------------------------------------------------------
// memory model: sparse word addressed store
// --------------------------------------------------------------------------
class MemModel {
  public:
    void write(uint64_t addr, uint64_t data, unsigned bytes, uint64_t strb) {
        const uint64_t base = addr & ~7ull;
        uint32_t lo, hi;
        read64(base, lo, hi);
        for (unsigned i = 0; i < 8; ++i) {
            const uint64_t byte_addr = base + i;
            if (byte_addr < addr || byte_addr >= addr + bytes) continue;
            if (strb && !((strb >> i) & 1)) continue;
            const uint32_t byte_val = (uint32_t)((data >> (8 * (byte_addr - addr))) & 0xff);
            uint32_t& word = ((byte_addr & 4) ? hi : lo);
            const int sh = 8 * (byte_addr & 3);
            word = (word & ~(0xffu << sh)) | (byte_val << sh);
        }
        words_[base] = lo;
        if (hi) words_[base + 4] = hi;
    }

    uint32_t read32(uint64_t addr) const {
        const auto it = words_.find(addr & ~3ull);
        return it == words_.end() ? 0u : it->second;
    }

    uint64_t read64(uint64_t addr) const { return read32(addr) | ((uint64_t)read32(addr + 4) << 32); }

    void load_image(const std::string& path, uint64_t base) {
        FILE* fh = std::fopen(path.c_str(), "rb");
        if (!fh) return;
        std::fseek(fh, 0, SEEK_END);
        const long size = std::ftell(fh);
        std::fseek(fh, 0, SEEK_SET);
        std::vector<uint8_t> buf((size_t)size);
        if (size > 0 && std::fread(buf.data(), 1, (size_t)size, fh) != (size_t)size) {
            buf.clear();
        }
        std::fclose(fh);
        for (size_t i = 0; i + 4 <= buf.size(); i += 4) {
            const uint32_t w = (uint32_t)buf[i] | ((uint32_t)buf[i + 1] << 8) |
                               ((uint32_t)buf[i + 2] << 16) | ((uint32_t)buf[i + 3] << 24);
            words_[base + i] = w;
        }
    }

    const std::map<uint64_t, uint32_t>& words() const { return words_; }

  private:
    void read64(uint64_t base, uint32_t& lo, uint32_t& hi) const {
        lo = read32(base);
        hi = read32(base + 4);
    }
    std::map<uint64_t, uint32_t> words_;
};

}  // namespace gpufuzz
#endif  // GPUFUZZ_RT_H
