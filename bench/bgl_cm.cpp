// Experimental BGL row codec; not a zform format (see bench/bgl_cm.py). Every per-row column goes
// through one binary arithmetic coder driven by context-mixing models, in row order: node,
// template selector, head tuple, clock delta, then the selected template's numbers.
// Build: c++ -O2 -std=c++17 bench/bgl_cm.cpp -o /tmp/bgl_cm
// Usage: bgl_cm pack ARR_DIR OUT.cm | bgl_cm unpack IN.cm ARR_DIR
// Both directions read nodecard/nodemid/noderack/nodepos.u32 (dictionary ids per node) and
// lanes.u32 (numbers per selector; selector 0 is a fallback row with none). Pack reads and unpack
// writes the rows: node.u32 sel.u32 head.u32 ticks.i64 and nums.i64 (row-major numbers).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;

static void die(const char *m) { fprintf(stderr, "bgl_cm: %s\n", m); exit(1); }
template <class T> vector<T> load(const string &p) {
  FILE *f = fopen(p.c_str(), "rb"); if (!f) die(p.c_str());
  fseek(f, 0, SEEK_END); size_t n = ftell(f); fseek(f, 0, SEEK_SET);
  vector<T> v(n / sizeof(T)); if (fread(v.data(), sizeof(T), v.size(), f) != v.size()) die(p.c_str()); fclose(f); return v;
}
template <class T> void save(const string &p, const vector<T> &v) {
  FILE *f = fopen(p.c_str(), "wb"); if (!f || fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) die(p.c_str()); fclose(f);
}
static inline uint32_t hsh(uint64_t a, uint64_t b) { uint64_t x = (a + 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull ^ b * 0x94D049BB133111EBull; x ^= x >> 29; x *= 0xD6E8FEB86659FD93ull; return x >> 32; }
static inline int nbits(uint64_t x) { return x ? 64 - __builtin_clzll(x) : 0; }
// log-scale bucket keeping m bits below the leading one
static inline uint32_t q(uint64_t x, int m) { int b = nbits(x); if (b <= m + 1) return x; return (b << m) | ((x >> (b - 1 - m)) & ((1u << m) - 1)); }
static inline int width(uint64_t n) { return n > 1 ? nbits(n - 1) : 0; } // bits for an index below n
template <class V, class T> static uint64_t indexOf(const V &v, T x) { return find(v.begin(), v.end(), x) - v.begin(); }
const uint32_t NONE = ~0u;

// Carryless binary arithmetic coder (lpaq style) with 16-bit probabilities.
struct Coder {
  bool enc = true; uint32_t x1 = 0, x2 = ~0u, x = 0; vector<uint8_t> buf; size_t pos = 0;
  uint8_t next() { return pos < buf.size() ? buf[pos++] : 0; }
  void start() { for (int k = 0; k < 4; k++) x = x << 8 | next(); }
  void flush() { for (int k = 0; k < 4; k++, x1 <<= 8) buf.push_back(x1 >> 24); }
  // Codes b (ignored when decoding) with P(1) = p / 65536 and returns the coded bit.
  int bit(int b, uint32_t p, double &cost) {
    uint32_t xm = x1 + (uint32_t)((uint64_t)(x2 - x1) * p >> 16);
    if (!enc) b = x <= xm;
    if (b) x2 = xm; else x1 = xm + 1;
    // Accounting does not affect the model; reuse exact costs for the 16-bit probabilities.
    static const auto costs = [] {
      array<double, 65536> table{};
      for (size_t i = 1; i < table.size(); i++) table[i] = -log2(i / 65536.0);
      return table;
    }();
    cost += costs[b ? p : 65536 - p];
    while ((x1 ^ x2) < 1u << 24) {
      if (enc) buf.push_back(x2 >> 24); else x = x << 8 | next();
      x1 <<= 8; x2 = x2 << 8 | 255;
    }
    return b;
  }
  uint64_t raw(uint64_t v, int n, double &cost) { uint64_t r = 0; for (int j = n - 1; j >= 0; j--) r = r << 1 | bit(v >> j & 1, 32768, cost); return r; }
} AC;

template <class T> static void put(T &slot, T got) { if (AC.enc && got != slot) die("model decoded a different value"); slot = got; }

static int TB = 22; // counter table bits, scaled down for small inputs
// Logistic mix of K hashed counters with adaptive rates; `sel` picks the weight set.
struct CM {
  // Keep both fields together without three padding bytes per entry in the large tables.
  struct __attribute__((packed)) Ctr { float p = 0.5f; uint8_t n = 0; };
  int K, tb, limit; float lr; vector<vector<Ctr>> tab; vector<float> w; float rate[256]; double cost = 0;
  CM(int K, int nsel, float lr, int limit = 60) : K(K), tb(TB), limit(limit), lr(lr), tab(K, vector<Ctr>(1u << TB)), w((size_t)nsel * (K + 1), 0.3f) {
    for (int i = 0; i < 256; i++) rate[i] = 1.0f / (i + 1.6f);
  }
  int code(int bit, const uint32_t *ctx, int sel) {
    Ctr *slot[32]; float st[33], dot = 0, *ww = &w[(size_t)sel * (K + 1)];
    // Issue the independent table requests before the logarithms serialize their use.
    for (int k = 0; k < K; k++) {
      slot[k] = &tab[k][hsh(ctx[k], k) >> (32 - tb)];
      __builtin_prefetch(slot[k], 1, 1);
    }
    for (int k = 0; k < K; k++) { float p = min(max(slot[k]->p, 0.0005f), 0.9995f); st[k] = logf(p / (1 - p)); }
    st[K] = 0.3f;
    for (int k = 0; k <= K; k++) dot += ww[k] * st[k];
    float p = min(max(1 / (1 + expf(-dot)), 0.0003f), 0.9997f);
    bit = AC.bit(bit, min(max((int)lrintf(p * 65536), 1), 65535), cost);
    float err = (bit - p) * lr;
    for (int k = 0; k <= K; k++) ww[k] += err * st[k];
    for (int k = 0; k < K; k++) { Ctr &s = *slot[k]; s.p += (bit - s.p) * rate[s.n]; if (s.n < limit) s.n++; }
    return bit;
  }
};

// The predictor lists are tiny; preserve tie order without allocating a merge buffer.
template <size_t N> static void rankScores(int (&order)[N], const array<float, N> &score) {
  for (size_t i = 0; i < N; i++) {
    size_t j = i;
    while (j && score[order[j - 1]] < score[i]) { order[j] = order[j - 1]; --j; }
    order[j] = i;
  }
}

struct Dict { vector<uint32_t> card, mid, rack, pos, lanes; };

// Node: recency rank of the midplane stream, then candidates within that stream, else an explicit
// card and node. Ids follow first appearance, so an unseen node is always the next id.
struct Node {
  static const int NC = 7;
  const Dict &D; int NM; vector<int> mtf; vector<uint32_t> last, succ, sp1, sp2, cardsucc; uint32_t cardFirst = NONE, nseen = 0;
  vector<int64_t> lastT; vector<uint16_t> hist; vector<uint8_t> seen; unordered_map<uint64_t, uint32_t> succt, byName;
  vector<vector<uint32_t>> cardsOf, nodesOn; int prank = 0, prank2 = 0; uint32_t pm = NONE;
  CM rank{7, 16, .005f}, within{7, 8, .005f}, esc{2, 4, .005f}; double rankraw = 0, rawc = 0;
  Node(const Dict &D) : D(D), NM(*max_element(D.mid.begin(), D.mid.end()) + 1), mtf(NM), last(NM, NONE), succ(D.card.size(), NONE),
      sp1(*max_element(D.pos.begin(), D.pos.end()) + 1, NONE), sp2(sp1), cardsucc(*max_element(D.card.begin(), D.card.end()) + 1, NONE),
      lastT(NM, 0), hist(NM, 0), seen(D.card.size(), 0), cardsOf(NM), nodesOn(cardsucc.size()) {
    for (int i = 0; i < NM; i++) mtf[i] = i;
    for (uint32_t x = 0; x < D.card.size(); x++) byName[(uint64_t)D.card[x] << 32 | D.pos[x]] = x;
  }
  uint32_t named(uint32_t card, uint32_t pos) { auto f = byName.find((uint64_t)card << 32 | pos); return f == byName.end() ? NONE : f->second; }
  // t, pt: template contexts; now: previous row's tick (the current one is coded later)
  uint32_t code(uint32_t n, uint32_t t, uint32_t pt, int64_t now) {
    if (pm != NONE) lastT[pm] = now;
    uint32_t c[8]; int r = 12, rt = indexOf(mtf, (int)D.mid[n]);
    for (int k = 0; k < 12; k++) {
      int cand = mtf[k]; uint32_t ln = last[cand];
      c[0] = k; c[1] = hsh(k, t); c[2] = hsh(k, hsh(prank, prank2)); c[3] = hsh(k, hsh(t, pt));
      c[4] = hsh(k, hsh(q(now - lastT[cand], 2), q(now - lastT[mtf[0]], 2)));
      c[5] = hsh(k, hsh(hist[cand], t)); c[6] = hsh(k, hsh(ln == NONE ? 0 : succ[ln] != NONE, prank));
      if (rank.code(k == rt, c, k)) { r = k; break; }
    }
    uint32_t m = r < 12 ? mtf[r] : AC.raw(D.mid[n], width(NM), rankraw);
    if (r == 12) r = indexOf(mtf, (int)m);
    uint32_t ln = last[m], s1 = ln == NONE ? NONE : succ[ln], s2 = s1 == NONE ? NONE : succ[s1], st = NONE, spa = NONE, spb = NONE, sc = NONE;
    if (ln != NONE) {
      auto it = succt.find((uint64_t)t << 32 | ln); if (it != succt.end()) st = it->second;
      if (sp1[D.pos[ln]] != NONE) spa = named(D.card[ln], sp1[D.pos[ln]]);
      if (sp2[D.pos[ln]] != NONE) spb = named(D.card[ln], sp2[D.pos[ln]]);
      // first position of the card that followed this card last time
      if (cardsucc[D.card[ln]] != NONE && cardFirst != NONE) sc = named(cardsucc[D.card[ln]], cardFirst);
    }
    uint32_t cands[NC] = {st, s1, spa, ln, s2, spb, sc}; int hit = NC;
    for (int k = 0; k < NC; k++) {
      bool dup = cands[k] == NONE; for (int j = 0; j < k; j++) dup |= cands[j] == cands[k];
      if (dup) continue;
      c[0] = k; c[1] = hsh(k, t); c[2] = hsh(k, hist[m]); c[3] = hsh(k, hsh(t, hist[m])); c[4] = hsh(k, hsh(t, pt));
      c[5] = hsh(k, hsh(r, hist[m] & 3)); c[6] = hsh(k, hsh(st == s1, hsh(s1 == ln, q(now - lastT[m], 2))));
      if (within.code(cands[k] == n, c, k)) { hit = k; n = cands[k]; break; }
    }
    if (hit == NC) {
      c[0] = 0; c[1] = t;
      if (esc.code(!seen[n], c, 0)) n = nseen;
      else {
        c[0] = 1; c[1] = hsh(t, 0); auto &cards = cardsOf[m];
        bool same = esc.code(ln != NONE && D.card[ln] == D.card[n], c, 1);
        auto &nodes = nodesOn[same ? D.card[ln] : cards[AC.raw(indexOf(cards, D.card[n]), width(cards.size()), rawc)]];
        n = nodes[AC.raw(indexOf(nodes, n), width(nodes.size()), rawc)];
      }
    }
    if (!seen[n]) {
      seen[n] = 1; nseen++; nodesOn[D.card[n]].push_back(n);
      if (indexOf(cardsOf[m], D.card[n]) == cardsOf[m].size()) cardsOf[m].push_back(D.card[n]);
    }
    if (ln != NONE && ln != n) {
      succ[ln] = n;
      if (D.card[ln] == D.card[n]) { uint32_t &a = sp1[D.pos[ln]]; if (a != D.pos[n]) sp2[D.pos[ln]] = a; a = D.pos[n]; }
      else { cardsucc[D.card[ln]] = D.card[n]; cardFirst = D.pos[n]; }
    }
    if (ln != NONE) succt[(uint64_t)t << 32 | ln] = n;
    hist[m] = (hist[m] << 3 | min(hit, 7)) & 0x1ff; last[m] = n; pm = m;
    mtf.erase(mtf.begin() + r); mtf.insert(mtf.begin(), m); prank2 = prank; prank = min(r, 12);
    return n;
  }
};

// Template selector given the node: hit flags over predictions ordered by recent success, then a
// 64-entry recency cache, else the selector in plain bits.
struct Sel {
  static const int NE = 9;
  const Dict &D; CM flag{7, NE + 1, .004f}, idx{3, 8, .004f}; double rawc = 0;
  unordered_map<uint64_t, uint32_t> s1, s2, s3, n1, n2, m1; vector<uint32_t> lastOnNode, last2OnNode, lastOnMid;
  array<float, NE> score{}; vector<uint32_t> cache; int hist = 0; uint32_t p1 = 0, p2 = 0, p3 = 0; bool any = false;
  Sel(const Dict &D) : D(D), lastOnNode(D.card.size(), NONE), last2OnNode(D.card.size(), NONE), lastOnMid(256, NONE) {}
  uint32_t code(uint32_t t, uint32_t nd, uint32_t pnd) {
    uint32_t ln = lastOnNode[nd], ln2 = last2OnNode[nd], lm = lastOnMid[D.mid[nd] & 255], E[NE], c[8]; bool have[NE] = {false};
    int rel = nd == pnd ? 0 : D.card[nd] == D.card[pnd] ? 1 : D.mid[nd] == D.mid[pnd] ? 2 : 3;
    auto get = [&](unordered_map<uint64_t, uint32_t> &m, uint64_t key, int e) { auto it = m.find(key); if (it != m.end()) { E[e] = it->second; have[e] = true; } };
    E[0] = p1; have[0] = any;
    get(s1, p1, 1); get(s2, (uint64_t)p2 << 32 | p1, 2); get(s3, ((uint64_t)p3 << 42) ^ ((uint64_t)p2 << 21) ^ p1, 3);
    if (ln != NONE) { E[4] = ln; have[4] = true; get(n1, ln, 5); get(n2, (uint64_t)ln2 << 32 | ln, 6); }
    if (lm != NONE) { E[7] = lm; have[7] = true; get(m1, (uint64_t)lm << 32 | p1, 8); }
    int order[NE]; for (int e = 0; e < NE; e++) order[e] = e;
    rankScores(order, score);
    int hit = NE + 1, tried = 0; uint32_t agree = 0; for (int e = 1; e < NE; e++) agree |= (have[e] && E[e] == E[0]) << e;
    for (int oi = 0; oi < NE; oi++) {
      int e = order[oi]; if (!have[e]) continue;
      bool dup = false; for (int g = 0; g < oi; g++) dup |= have[order[g]] && E[order[g]] == E[e];
      if (dup) continue;
      c[0] = hsh(e, tried); c[1] = hsh(hsh(e, tried), hist & 0xfff); c[2] = hsh(hsh(e, tried), E[e]); c[3] = hsh(hsh(e, E[e]), p1); c[4] = hsh(hsh(e, rel), hsh(hist & 0xff, agree));
      c[5] = hsh(hsh(e, E[e]), hsh(p1, p2)); c[6] = hsh(hsh(e, E[e]), hsh(rel, ln));
      int ok = flag.code(E[e] == t, c, min(tried, NE - 1)); tried++;
      if (ok) { hit = e; t = E[e]; break; }
    }
    if (hit > NE) {
      uint64_t at = indexOf(cache, t);
      c[0] = 1; c[1] = hsh(1, hist & 0xfff); c[2] = hsh(1, p1); c[3] = hsh(1, hsh(p1, p2)); c[4] = hsh(1, rel); c[5] = hsh(1, ln); c[6] = hsh(1, tried);
      if (flag.code(at < cache.size(), c, NE)) {
        hit = NE; uint32_t pre = 1;
        for (int b = 5; b >= 0; b--) { c[0] = pre; c[1] = hsh(pre, p1); c[2] = hsh(pre, hist & 0xff); pre = pre * 2 + idx.code(at >> b & 1, c, 5 - b); }
        t = cache[pre - 64];
      } else t = AC.raw(t, width(D.lanes.size()), rawc);
    }
    for (int e = 0; e < NE; e++) if (have[e]) score[e] += ((E[e] == t) - score[e]) * 0.02f;
    s1[p1] = t; s2[(uint64_t)p2 << 32 | p1] = t; s3[((uint64_t)p3 << 42) ^ ((uint64_t)p2 << 21) ^ p1] = t;
    if (ln != NONE) { n1[ln] = t; n2[(uint64_t)ln2 << 32 | ln] = t; }
    if (lm != NONE) m1[(uint64_t)lm << 32 | p1] = t;
    last2OnNode[nd] = ln; lastOnNode[nd] = t; lastOnMid[D.mid[nd] & 255] = t;
    uint64_t at = indexOf(cache, t); if (at < cache.size()) cache.erase(cache.begin() + at);
    cache.insert(cache.begin(), t); if (cache.size() > 64) cache.pop_back();
    hist = (hist << 4 | min(hit, 15)) & 0xffff; p3 = p2; p2 = p1; p1 = t; any = true;
    return t;
  }
};

// Head tuple: the last one seen with this template, else with this template on this node.
struct Head {
  int bits; CM hd{4, 4, .004f}; double rawc = 0; unordered_map<uint64_t, uint32_t> hlast;
  Head(uint32_t nheads) : bits(width(nheads)) {}
  uint32_t code(uint32_t h, uint32_t t, uint32_t nd, uint32_t ph) {
    auto it = hlast.find(t); uint32_t h0 = it == hlast.end() ? NONE : it->second, c[4];
    auto it1 = hlast.find((uint64_t)(nd + 1) << 32 | t); uint32_t h1 = it1 == hlast.end() ? NONE : it1->second;
    c[0] = hsh(t, 0); c[1] = hsh(t, h0); c[2] = hsh(h0, h1 == h0); c[3] = hsh(t, ph);
    if (hd.code(h == h0, c, 0)) h = h0;
    else {
      c[0] = hsh(t, 1); c[1] = hsh(h1, 1); c[2] = hsh(t, h1); c[3] = 7;
      h = hd.code(h1 != NONE && h == h1, c, 1) ? h1 : AC.raw(h, bits, rawc);
    }
    hlast[t] = h; hlast[(uint64_t)(nd + 1) << 32 | t] = h;
    return h;
  }
};

// Clock delta: 6-bit length, then mantissa bits whose contexts compare the coded prefix with
// predictions (recent gaps, and this midplane's last gap less the time since it spoke).
struct Clock {
  static const int NB = 12, NE = 6, DEPTH = 8;
  const Dict &D; CM cm{NB + NE, 64 + 41 * 20, .004f}; double rawc = 0;
  vector<int64_t> lastmid = vector<int64_t>(256, 0), gapmid = vector<int64_t>(256, 0); uint64_t pd[8] = {0}; uint32_t recentmid[8] = {0};
  Clock(const Dict &D) : D(D) {}
  int64_t code(int64_t tick, int64_t prev, bool first, uint32_t n, uint32_t pn, uint32_t t, uint32_t pt) {
    if (first) return AC.raw(tick, 64, rawc);
    uint64_t dt = tick - prev; if (AC.enc && dt >> 63) die("ticks must not decrease");
    uint32_t m = D.mid[n], base[NB], c[NB + NE];
    int r = n == pn ? 0 : D.card[n] == D.card[pn] ? 1 : m == D.mid[pn] ? 2 : D.rack[n] == D.rack[pn] ? 3 : 4;
    uint64_t sm = prev - lastmid[m & 255];
    int act = 0; for (int a = 0; a < 8; a++) { bool dup = false; for (int b = 0; b < a; b++) dup |= recentmid[b] == recentmid[a]; act += !dup; }
    uint64_t mn3 = min(pd[0], min(pd[1], pd[2])), mn5 = min(mn3, min(pd[3], pd[4]));
    uint64_t s5[5] = {pd[0], pd[1], pd[2], pd[3], pd[4]}; sort(s5, s5 + 5); uint64_t med5 = s5[2];
    uint64_t gm = gapmid[m & 255], estream = gm > sm ? gm - sm : 0;
    uint64_t E[NE] = {mn3, pd[0], med5, estream, pd[1], mn5};
    uint32_t length_ctx[NE];
    for (int e = 0; e < NE; e++) length_ctx[e] = hsh(q(E[e], 3), NB + e);
    base[0] = 0; base[1] = q(pd[0], 4); base[2] = hsh(q(pd[0], 3), q(pd[1], 3)); base[3] = hsh(t, r);
    base[4] = hsh(hsh(t, pt), q(pd[0], 2)); base[5] = hsh(r, q(pd[0], 3)); base[6] = hsh(act, q(pd[0], 3));
    base[7] = hsh(q(sm, 3), act); base[8] = hsh(q(mn3, 4), q(pd[0] - mn3, 2)); base[9] = hsh(q(gm, 3), hsh(q(sm, 3), r == 0));
    base[10] = hsh(q(med5, 4), act); base[11] = hsh(hsh(q(pd[0], 2), q(pd[1], 2)), hsh(q(pd[2], 2), r));
    uint32_t prefix = 1; uint64_t d = 0; // d: decoded high bits, i.e. dt >> shift
    auto bit = [&](int b, int sel, int shift, bool isnb) {
      for (int k = 0; k < NB; k++) c[k] = hsh(base[k], prefix);
      for (int e = 0; e < NE; e++) {
        if (isnb) { c[NB + e] = hsh(length_ctx[e], prefix); continue; }
        int64_t df = max<int64_t>(-3, min<int64_t>(3, (int64_t)d - (int64_t)(E[e] >> shift)));
        uint32_t nx = shift >= 4 ? (E[e] >> (shift - 4)) & 15 : (E[e] << (4 - shift)) & 15;
        c[NB + e] = hsh(hsh(df + 3, nx), hsh(NB + e, sel));
      }
      return cm.code(b, c, sel);
    };
    int nbt = nbits(dt), nb = 0;
    for (int j = 5; j >= 0; j--) { int b = bit(nbt >> j & 1, 5 - j, 0, true); prefix = prefix * 2 + b; nb = nb * 2 + b; }
    prefix = 64 + nb; d = nb > 0;
    for (int j = nb - 2, k = 0; j >= 0; j--, k++) {
      if (k >= DEPTH) { d = d << (j + 1) | AC.raw(dt, j + 1, rawc); break; }
      int b = bit(dt >> j & 1, 64 + min(nb, 40) * 20 + k, j + 1, false); prefix = prefix * 2 + b; d = d * 2 + b;
    }
    memmove(pd + 1, pd, 7 * sizeof *pd); pd[0] = d;
    gapmid[m & 255] = prev + d - lastmid[m & 255]; lastmid[m & 255] = prev + d;
    memmove(recentmid + 1, recentmid, 7 * sizeof *recentmid); recentmid[0] = m;
    return prev + d;
  }
};

// Template numbers, per lane: exact-hit flags over predictions ordered by recent success, a
// recent-value cache, else the value bitwise with contexts comparing its prefix with predictions.
struct Num {
  static const int NE = 19, CACHE = 64;
  struct Lane { int hist = 0, res = 0; int64_t delta = 0, prev2 = 0; array<float, NE> err, score; vector<int64_t> cache; Lane() { err.fill(8); score.fill(0); } };
  struct Tpl { uint32_t rows = 0, pnd = 0; vector<Lane> lane; vector<int64_t> prev; };
  const Dict &D; vector<Tpl> tpl; vector<uint64_t> base; CM flag{6, NE + 1, .004f}, res{7, 96, .004f}, idx{3, 6, .004f};
  unordered_map<uint64_t, array<int64_t, 2>> lastNode; unordered_map<uint64_t, int64_t> pairDelta, lastMid, seq1, seq2, assoc, assoc2, nsucc;
  Num(const Dict &D) : D(D), tpl(D.lanes.size()), base(D.lanes.size(), 0) {
    for (size_t s = 1; s < base.size(); s++) base[s] = base[s - 1] + D.lanes[s - 1];
  }
  void code(uint32_t s, uint32_t nd, int64_t *v) {
    uint32_t P = D.lanes[s], c[8]; if (!P) return;
    Tpl &T = tpl[s]; if (T.lane.empty()) { T.lane.resize(P); T.prev.assign(P, 0); }
    uint32_t k = T.rows, pnd = k ? T.pnd : nd;
    int rel = nd == pnd ? 0 : D.card[nd] == D.card[pnd] ? 1 : D.mid[nd] == D.mid[pnd] ? 2 : 3;
    for (uint32_t j = 0; j < P; j++) {
      Lane &Ln = T.lane[j]; uint64_t lane = base[s] + j; int64_t x = v[j], prev = k ? T.prev[j] : 0; int pr = j ? T.lane[j - 1].res : 99;
      uint64_t kn = lane << 32 | nd, kp = (lane << 24 | D.pos[pnd]) << 20 | D.pos[nd] << 2 | min(rel, 3), km = lane << 32 | D.mid[nd];
      auto in = lastNode.find(kn); bool known = in != lastNode.end(); auto ip = pairDelta.find(kp); auto im = lastMid.find(km);
      int64_t E[NE]; bool have[NE] = {false};
      E[0] = prev; have[0] = k > 0;
      if (known) { E[1] = in->second[0]; have[1] = true; E[2] = in->second[1]; have[2] = in->second[1] != in->second[0]; }
      if (k && ip != pairDelta.end()) { E[3] = prev + ip->second; have[3] = true; }
      if (k > 1) { E[4] = prev + Ln.delta; have[4] = true; }
      for (uint32_t L = 1; L <= 4; L++) if (j >= L) { E[4 + L] = v[j - L] + (k ? T.prev[j] - T.prev[j - L] : 0); have[4 + L] = true; }
      if (im != lastMid.end()) { E[9] = im->second; have[9] = true; }
      for (uint32_t L = 1; L <= 3; L++) if (j >= 2 * L) { E[9 + L] = 2 * v[j - L] - v[j - 2 * L]; have[9 + L] = true; }
      if (j) { E[13] = v[j - 1]; have[13] = true; }
      // successors of the lane's last one or two values, of the previous parameter's value (alone
      // and with the lane's last value), and of this node's last value
      auto look = [&](unordered_map<uint64_t, int64_t> &m, uint64_t key, int e) { auto it = m.find(key); if (it != m.end()) { E[e] = it->second; have[e] = true; } };
      uint64_t up = prev, u2 = Ln.prev2, ul = j ? v[j - 1] : 0, un = known ? in->second[0] : 0;
      uint64_t k1 = hsh(lane, up) | (uint64_t)hsh(up, 5) << 32, k2 = hsh(lane ^ u2 * 31, up) | (uint64_t)hsh(u2, up) << 32;
      uint64_t ka = hsh(lane, ul) | (uint64_t)hsh(ul, 9) << 32, ka2 = hsh(lane ^ up * 131, ul) | (uint64_t)hsh(ul, up) << 32, kns = hsh(lane, un) | (uint64_t)hsh(un, 13) << 32;
      if (k) { look(seq1, k1, 14); if (k > 1) look(seq2, k2, 15); }
      if (j) { look(assoc, ka, 16); if (k) look(assoc2, ka2, 18); }
      if (known) look(nsucc, kns, 17);
      int order[NE]; for (int e = 0; e < NE; e++) order[e] = e;
      rankScores(order, Ln.score);
      int hit = NE + 1, tried = 0;
      for (int oi = 0; oi < NE; oi++) {
        int e = order[oi]; if (!have[e]) continue;
        bool dup = false; for (int g = 0; g < oi; g++) dup |= have[order[g]] && E[order[g]] == E[e];
        if (dup) continue;
        uint32_t L = hsh(lane, e);
        c[0] = L; c[1] = hsh(L, Ln.hist & 0xfff); c[2] = hsh(L, hsh(pr, Ln.hist & 0xf)); c[3] = hsh(L, hsh(rel, Ln.hist & 0xff));
        c[4] = hsh(L, hsh(tried, pr)); c[5] = hsh(hsh(e, tried), hsh(Ln.hist & 0xff, P));
        int ok = flag.code(E[e] == x, c, min(tried, NE - 1)); tried++;
        if (ok) { hit = e; x = E[e]; break; }
      }
      auto &ch = Ln.cache;
      if (hit > NE) {
        uint64_t at = indexOf(ch, x); uint32_t L = hsh(lane, 55);
        c[0] = L; c[1] = hsh(L, Ln.hist & 0xfff); c[2] = hsh(L, pr); c[3] = hsh(L, rel); c[4] = hsh(L, ch.size()); c[5] = hsh(P, Ln.hist & 0xff);
        if (flag.code(at < ch.size(), c, NE)) {
          hit = NE; uint32_t pre = 1;
          for (int b = 5; b >= 0; b--) { c[0] = hsh(L, pre); c[1] = hsh(pre, 0); c[2] = hsh(hsh(L, pre), Ln.hist & 0xf); pre = pre * 2 + idx.code(at >> b & 1, c, 5 - b); }
          x = ch[pre - CACHE];
        }
      }
      if (hit > NE) x = value(x, lane, P, j, pr, E, have, Ln);
      for (int e = 0; e < NE; e++) if (have[e]) { int64_t df = x - E[e]; Ln.err[e] += (nbits(df < 0 ? -(uint64_t)df : df) - Ln.err[e]) * 0.1f; Ln.score[e] += ((E[e] == x) - Ln.score[e]) * 0.05f; }
      Ln.hist = (Ln.hist & ~0xfff) | ((Ln.hist << 4 | min(hit, 15)) & 0xfff); Ln.res = hit;
      if (k) { Ln.delta = x - prev; pairDelta[kp] = x - prev; }
      if (!known) lastNode[kn] = {x, x}; else if (in->second[0] != x) in->second = {x, in->second[0]};
      lastMid[km] = x;
      if (k) { seq1[k1] = x; if (k > 1) seq2[k2] = x; }
      if (j) { assoc[ka] = x; if (k) assoc2[ka2] = x; }
      if (known) nsucc[kns] = x;
      Ln.prev2 = prev;
      uint64_t at = indexOf(ch, x); if (at < ch.size()) ch.erase(ch.begin() + at);
      ch.insert(ch.begin(), x); if (ch.size() > CACHE) ch.pop_back();
      put(v[j], x);
    }
    T.prev.assign(v, v + P); T.pnd = nd; T.rows++;
  }
  int64_t value(uint64_t a, uint64_t lane, uint32_t P, uint32_t j, int pr, const int64_t *E, const bool *have, Lane &Ln) {
    int best = -1; for (int e = 0; e < NE; e++) if (have[e] && (best < 0 || Ln.err[e] < Ln.err[best])) best = e;
    uint64_t es[3] = {best < 0 ? 0 : (uint64_t)E[best], have[0] ? (uint64_t)E[0] : 0, have[1] ? (uint64_t)E[1] : 0}, d = 0; // d: decoded high bits
    uint32_t L = hsh(lane, 77), prefix = 1, c[7]; int pnb = Ln.hist >> 12;
    auto bit = [&](int b, int sel, int shift, bool isnb) {
      uint32_t lp = hsh(L, prefix); c[0] = lp; c[1] = hsh(lp, pnb); c[2] = hsh(lp, pr); c[3] = hsh(hsh(P, j), prefix);
      for (int g = 0; g < 3; g++) {
        if (isnb) { c[4 + g] = hsh(lp, hsh(g, q(es[g], 2))); continue; }
        int64_t df = max<int64_t>(-3, min<int64_t>(3, (int64_t)d - (int64_t)(es[g] >> shift)));
        uint32_t nx = shift >= 3 ? (es[g] >> (shift - 3)) & 7 : (es[g] << (3 - shift)) & 7;
        c[4 + g] = hsh(hsh(L, g), hsh(df + 3, hsh(nx, min(shift, 20))));
      }
      return res.code(b, c, sel);
    };
    int nbt = nbits(a), nb = 0;
    for (int b = 5; b >= 0; b--) { int y = bit(nbt >> b & 1, 1 + (5 - b), 0, true); prefix = prefix * 2 + y; nb = nb * 2 + y; }
    prefix = 256 + nb; d = nb > 0;
    for (int b = nb - 2, k = 0; b >= 0; b--, k++) {
      if (k >= 12) prefix = 100000 + b;
      int y = bit(a >> b & 1, 8 + min(k, 15) + 16 * min(nb / 8, 4), b + 1, false); d = d * 2 + y;
      if (k < 12) prefix = prefix * 2 + y;
    }
    int64_t x = d, r = best < 0 ? x : x - E[best];
    Ln.hist = (Ln.hist & 0xfff) | min(nbits(r < 0 ? -(uint64_t)r : r), 63) << 12;
    return x;
  }
};

struct Rows { vector<uint32_t> node, sel, head; vector<int64_t> T, nums; };

static void run(Rows &R, const Dict &D, uint32_t nheads) {
  size_t N = R.node.size(), at = 0; TB = min(22, nbits(N) + 5);
  Node node(D); Sel sel(D); Head head(nheads); Clock clock(D); Num num(D);
  for (size_t i = 0; i < N; i++) {
    uint32_t pn = i ? R.node[i - 1] : 0, ps = i ? R.sel[i - 1] : 0;
    put(R.node[i], node.code(R.node[i], ps, ps, i ? R.T[i - 1] : 0));
    uint32_t n = R.node[i];
    put(R.sel[i], sel.code(R.sel[i], n, i ? pn : n));
    uint32_t s = R.sel[i]; if (s >= D.lanes.size()) die("selector out of range");
    put(R.head[i], head.code(R.head[i], s, n, i ? R.head[i - 1] : 0));
    put(R.T[i], clock.code(R.T[i], i ? R.T[i - 1] : 0, i == 0, n, pn, s, ps));
    uint32_t P = D.lanes[s];
    if (!AC.enc) R.nums.resize(at + P); else if (at + P > R.nums.size()) die("nums.i64 too short");
    num.code(s, n, P ? R.nums.data() + at : nullptr);
    at += P;
  }
  if (at != R.nums.size()) die("nums.i64 length does not match the selectors");
  double node_esc = node.esc.cost + node.rawc, total = 0;
  pair<const char *, double> parts[] = {
      {"node.rank", node.rank.cost + node.rankraw}, {"node.within", node.within.cost}, {"node.escape", node_esc},
      {"selector.flags", sel.flag.cost}, {"selector.cache", sel.idx.cost}, {"selector.raw", sel.rawc},
      {"head", head.hd.cost + head.rawc}, {"clock.cm", clock.cm.cost}, {"clock.raw", clock.rawc},
      {"nums.flags", num.flag.cost}, {"nums.cache", num.idx.cost}, {"nums.value", num.res.cost}};
  printf("{\"rows\": %zu, \"model_mb\": {", N);
  for (auto &p : parts) { printf("\"%s\": %.6f, ", p.first, p.second / 8e6); total += p.second; }
  printf("\"total\": %.6f}, ", total / 8e6);
}

static Dict dict(const string &dir) {
  Dict D{load<uint32_t>(dir + "/nodecard.u32"), load<uint32_t>(dir + "/nodemid.u32"), load<uint32_t>(dir + "/noderack.u32"),
         load<uint32_t>(dir + "/nodepos.u32"), load<uint32_t>(dir + "/lanes.u32")};
  if (D.card.empty() || D.lanes.empty()) die("empty dictionary");
  return D;
}

int main(int argc, char **argv) {
  if (argc != 4) die("usage: bgl_cm pack ARR_DIR OUT.cm | bgl_cm unpack IN.cm ARR_DIR");
  string mode = argv[1];
  if (mode == "pack") {
    string dir = argv[2]; Dict D = dict(dir);
    Rows R{load<uint32_t>(dir + "/node.u32"), load<uint32_t>(dir + "/sel.u32"), load<uint32_t>(dir + "/head.u32"), load<int64_t>(dir + "/ticks.i64"), load<int64_t>(dir + "/nums.i64")};
    size_t N = R.node.size(); if (!N || R.sel.size() != N || R.head.size() != N || R.T.size() != N) die("row arrays differ in length");
    for (auto n : R.node) if (n >= D.card.size()) die("node id out of range");
    uint64_t hdr[2] = {N, (uint64_t)*max_element(R.head.begin(), R.head.end()) + 1};
    run(R, D, hdr[1]); AC.flush();
    FILE *f = fopen(argv[3], "wb"); if (!f) die(argv[3]);
    fwrite(hdr, 8, 2, f); fwrite(AC.buf.data(), 1, AC.buf.size(), f); fclose(f);
    printf("\"stream_bytes\": %zu}\n", AC.buf.size() + sizeof hdr);
  } else if (mode == "unpack") {
    string dir = argv[3]; Dict D = dict(dir); auto in = load<uint8_t>(argv[2]); uint64_t hdr[2];
    if (in.size() < sizeof hdr) die("truncated stream");
    memcpy(hdr, in.data(), sizeof hdr);
    AC.enc = false; AC.buf.assign(in.begin() + sizeof hdr, in.end()); AC.start();
    Rows R; R.node.resize(hdr[0]); R.sel.resize(hdr[0]); R.head.resize(hdr[0]); R.T.resize(hdr[0]);
    run(R, D, hdr[1]);
    save(dir + "/node.u32", R.node); save(dir + "/sel.u32", R.sel); save(dir + "/head.u32", R.head); save(dir + "/ticks.i64", R.T); save(dir + "/nums.i64", R.nums);
    printf("\"stream_bytes\": %zu}\n", in.size());
  } else die("mode must be pack or unpack");
}
