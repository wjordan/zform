// Manual benchmark codec for ordered integer clocks, with source-change
// context.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <zstd.h>
struct Model {
  std::vector<std::array<unsigned, 2>> p;
  unsigned limit;
  Model(unsigned lim) : p(512 * 256), limit(lim) {
    for (auto &v : p)
      v = {1, 1};
  }
  void update(unsigned i, unsigned bit) {
    auto &v = p[i];
    v[bit]++;
    if (v[0] + v[1] >= limit) {
      v[0] = (v[0] + 1) / 2;
      v[1] = (v[1] + 1) / 2;
    }
  }
};
struct Bits {
  std::vector<uint8_t> data;
  size_t n = 0, pos = 0;
  void put(bool b) {
    if (n % 8 == 0)
      data.push_back(0);
    data.back() |= unsigned(b) << (7 - n % 8);
    n++;
  }
  unsigned get() {
    if (pos >= n) {
      pos++;
      return 0;
    }
    size_t at = pos++;
    return (data[at / 8] >> (7 - (at % 8))) & 1;
  }
};
struct Coder {
  uint32_t lo = 0, hi = ~0u, code = 0;
  size_t pending = 0;
  Bits bits;
  Model model{64};
  void emit(unsigned b) {
    bits.put(b);
    while (pending) {
      bits.put(!b);
      pending--;
    }
  }
  void encode(unsigned symbol, unsigned ctx) {
    unsigned node = 1;
    for (int shift = 7; shift >= 0; shift--) {
      unsigned index = ctx * 256 + node, bit = (symbol >> shift) & 1;
      auto p = model.p[index];
      uint32_t mid = lo + ((uint64_t(hi) - lo + 1) * p[0] / (p[0] + p[1])) - 1;
      if (bit)
        lo = mid + 1;
      else
        hi = mid;
      while (true) {
        if (hi < 0x80000000u)
          emit(0);
        else if (lo >= 0x80000000u) {
          emit(1);
          lo -= 0x80000000u;
          hi -= 0x80000000u;
        } else if (lo >= 0x40000000u && hi < 0xc0000000u) {
          pending++;
          lo -= 0x40000000u;
          hi -= 0x40000000u;
        } else
          break;
        lo <<= 1;
        hi = (hi << 1) | 1;
      }
      model.update(index, bit);
      node = node * 2 + bit;
    }
  }
  void finish() {
    pending++;
    emit(lo < 0x40000000u ? 0 : 1);
  }
  void start_decode(std::vector<uint8_t> b) {
    bits.data = std::move(b);
    bits.n = bits.data.size() * 8;
    for (int i = 0; i < 32; i++)
      code = (code << 1) | bits.get();
  }
  unsigned decode(unsigned ctx) {
    unsigned node = 1, symbol = 0;
    for (int shift = 7; shift >= 0; shift--) {
      unsigned index = ctx * 256 + node;
      auto p = model.p[index];
      uint32_t mid = lo + ((uint64_t(hi) - lo + 1) * p[0] / (p[0] + p[1])) - 1;
      unsigned bit = code > mid;
      symbol = (symbol << 1) | bit;
      if (bit)
        lo = mid + 1;
      else
        hi = mid;
      while (true) {
        if (hi < 0x80000000u) {
        } else if (lo >= 0x80000000u) {
          code -= 0x80000000u;
          lo -= 0x80000000u;
          hi -= 0x80000000u;
        } else if (lo >= 0x40000000u && hi < 0xc0000000u) {
          code -= 0x40000000u;
          lo -= 0x40000000u;
          hi -= 0x40000000u;
        } else
          break;
        lo <<= 1;
        hi = (hi << 1) | 1;
        code = (code << 1) | bits.get();
      }
      model.update(index, bit);
      node = node * 2 + bit;
    }
    return symbol;
  }
};
template <class T> std::vector<T> load(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open input");
  f.seekg(0, std::ios::end);
  size_t n = f.tellg();
  f.seekg(0);
  if (n % sizeof(T))
    throw std::runtime_error("bad input length");
  std::vector<T> v(n / sizeof(T));
  f.read((char *)v.data(), n);
  return v;
}
void put(std::ostream &o, uint64_t v, int n) {
  for (int i = 0; i < n; i++)
    o.put(v >> (8 * i));
}
uint64_t get(std::istream &i, int n) {
  uint64_t v = 0;
  for (int k = 0; k < n; k++) {
    int c = i.get();
    if (c < 0)
      throw std::runtime_error("truncated stream");
    v |= uint64_t(c) << (8 * k);
  }
  return v;
}
void blob(std::ostream &o, const std::vector<uint8_t> &v) {
  put(o, v.size(), 4);
  o.write((const char *)v.data(), v.size());
}
std::vector<uint8_t> blob(std::istream &i) {
  auto n = get(i, 4);
  if (n > 4000000)
    throw std::runtime_error("oversized block");
  std::vector<uint8_t> v(n);
  i.read((char *)v.data(), n);
  if (!i)
    throw std::runtime_error("truncated block");
  return v;
}
void varint(std::vector<uint8_t> &o, uint64_t v) {
  while (v >= 128) {
    o.push_back((v & 127) | 128);
    v >>= 7;
  }
  o.push_back(v);
}
uint64_t read_varint(const std::vector<uint8_t> &v, size_t &pos) {
  uint64_t x = 0;
  for (int s = 0; s < 64; s += 7) {
    if (pos >= v.size())
      throw std::runtime_error("truncated varint");
    auto b = v[pos++];
    x |= uint64_t(b & 127) << s;
    if (b < 128)
      return x;
  }
  throw std::runtime_error("bad varint");
}
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::runtime_error("Usage: bgl_context pack TICKS NODES OUTPUT | "
                               "unpack ARCHIVE NODES TICKS");
    auto nodes = load<uint32_t>(argv[3]);
    std::ofstream out(argv[4], std::ios::binary);
    if (!out)
      throw std::runtime_error("cannot open output");
    if (std::string(argv[1]) == "pack") {
      auto ticks = load<uint64_t>(argv[2]);
      if (ticks.size() != nodes.size())
        throw std::runtime_error("row count mismatch");
      out.write("BGC1", 4);
      put(out, ticks.size(), 8);
      for (size_t start = 0; start < ticks.size(); start += 200000) {
        size_t n = std::min(size_t(200000), ticks.size() - start);
        put(out, ticks[start], 8);
        Coder coder;
        Bits low;
        std::vector<uint8_t> tail;
        uint64_t previous = ticks[start], previous_high = 0;
        for (size_t j = 0; j < n; j++) {
          size_t i = start + j;
          if (ticks[i] < previous)
            throw std::runtime_error("clock is not ordered");
          auto d = ticks[i] - previous;
          previous = ticks[i];
          auto h = d >> 10;
          unsigned ctx = std::min(previous_high, uint64_t(255)) * 2 +
                         (i > 0 && nodes[i] == nodes[i - 1]);
          coder.encode(std::min(h, uint64_t(255)), ctx);
          previous_high = h;
          if (h >= 255)
            varint(tail, h - 255);
          for (int bit = 0; bit < 10; bit++)
            low.put((d >> bit) & 1);
        }
        coder.finish();
        blob(out, low.data);
        blob(out, coder.bits.data);
        put(out, tail.size(), 4);
        std::vector<uint8_t> c(ZSTD_compressBound(tail.size()));
        auto len =
            ZSTD_compress(c.data(), c.size(), tail.data(), tail.size(), 9);
        if (ZSTD_isError(len))
          throw std::runtime_error("zstd encode failed");
        c.resize(len);
        blob(out, c);
      }
    } else if (std::string(argv[1]) == "unpack") {
      std::ifstream in(argv[2], std::ios::binary);
      char magic[4] = {};
      if (!in.read(magic, 4) || std::memcmp(magic, "BGC1", 4))
        throw std::runtime_error("bad magic");
      auto count = get(in, 8);
      if (count != nodes.size())
        throw std::runtime_error("row count mismatch");
      for (size_t start = 0; start < count; start += 200000) {
        size_t n = std::min(size_t(200000), count - start);
        uint64_t tick = get(in, 8), previous_high = 0;
        Bits low;
        low.data = blob(in);
        low.n = low.data.size() * 8;
        if (low.n < n * 10)
          throw std::runtime_error("truncated low bits");
        Coder coder;
        coder.start_decode(blob(in));
        auto size = get(in, 4);
        if (size > n * 10)
          throw std::runtime_error("oversized tail");
        std::vector<uint8_t> tail(size);
        auto c = blob(in);
        if (ZSTD_decompress(tail.data(), size, c.data(), c.size()) != size)
          throw std::runtime_error("zstd decode failed");
        size_t pos = 0;
        for (size_t j = 0; j < n; j++) {
          size_t i = start + j;
          unsigned ctx = std::min(previous_high, uint64_t(255)) * 2 +
                         (i > 0 && nodes[i] == nodes[i - 1]);
          uint64_t h = coder.decode(ctx);
          if (h == 255)
            h += read_varint(tail, pos);
          previous_high = h;
          uint64_t d = h << 10;
          for (int bit = 0; bit < 10; bit++)
            d |= uint64_t(low.get()) << bit;
          tick += d;
          put(out, tick, 8);
        }
        if (pos != tail.size())
          throw std::runtime_error("unused tail values");
      }
      if (in.peek() != EOF)
        throw std::runtime_error("trailing bytes");
    } else
      throw std::runtime_error("unknown mode");
    out.flush();
    return !out;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
