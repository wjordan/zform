"""Ideal code length of HDFS.log under a block-trace model (research estimator, not a codec).

HDFS.log is sorted by locale collation (alphanumerics only, case-folded), so a decoder can
rebuild line order by sorting and the coder may emit lines in any order. This groups lines by
block id, orders blocks by their java.util.Random index, and codes each block's trace with
adaptive models conditioned on the block's history, the inferred logging node and per-node
clocks (thread ids, ephemeral ports). Datanode deletions form a separate node-major stream.
Costs are -log2 p of causal adaptive models; an arithmetic coder would land within ~1%.

    python3 bench/hdfs_trace.py HDFS.log            # whole file
    WIN=60000 python3 bench/hdfs_trace.py HDFS.log  # reset entity state every 60k rows
    RESID=1 ...  # also size the predictors' residual streams under plain LZMA2
    SL=1 ...     # keyed previous-row references only: no per-node clocks or port counters
"""
import os, sys, re, bisect, calendar, time, math
from collections import defaultdict

# Ideal code-length estimate for HDFS in block-major order.
L2 = math.log2
MASK = (1 << 48) - 1; MULT = 0x5DEECE66D; ADD = 0xB
def out(s):
    n = (s * MULT + ADD) & MASK
    return ((((s >> 16) << 32) + ((n >> 16) ^ (1 << 31)) - (1 << 31)) + (1 << 63)) % (1 << 64) - (1 << 63)
def state_of(v):
    low = v & 0xffffffff; high = ((v >> 32) + (low >> 31)) & 0xffffffff
    t = ((low << 16) - ((high << 16) * MULT + ADD)) & MASK
    for k in range(-1, 6):
        base = t + (k << 48); x = -(-max(base, 0) // MULT)
        if x >> 16: break
        if x * MULT < base + (1 << 16):
            s = high << 16 | x
            return s if out(s) == v else None
    return None
JUMP = []
a, c = MULT, ADD
for i in range(48):
    JUMP.append((a, c)); c = (c * (a + 1)) & MASK; a = (a * a) & MASK
def dist(f, t):
    st, steps = f, 0
    for i, (a, c) in enumerate(JUMP):
        if st == t: break
        if (st ^ t) >> i & 1:
            st = (st * a + c) & MASK; steps |= 1 << i
    return steps - (1 << 48) if steps >> 47 else steps

class Cat:
    """Adaptive categorical with context and backoff to an order-0 model, escape to `newcost`."""
    def __init__(s, k=0.4):
        s.c = defaultdict(lambda: defaultdict(int)); s.n = defaultdict(int)
        s.c0 = defaultdict(int); s.n0 = 0; s.u0 = 0; s.k = k
    def cost(s, ctx, sym, newcost):
        if getattr(s, 'name', None): s.lastkey = (s.name, None if s.name == 'tplM' else CUR[0], WI[0]); LOG[s.lastkey].append(sym); NEWC[s.lastkey][sym] = newcost
        # order-0 with PPM-C style escape
        c0 = s.c0.get(sym, 0)
        esc0 = (s.u0 + 1) / (s.n0 + s.u0 + 1)
        p0 = (c0 / (s.n0 + s.u0 + 1)) if c0 else 0.0
        if not c0: p0 = esc0 * 2 ** -newcost
        cc = s.c[ctx]; n = s.n[ctx]; u = len(cc)
        k = s.k * (u + 1)
        p = (cc.get(sym, 0) + k * p0) / (n + k)
        if not c0: s.u0 += 1
        s.c0[sym] = c0 + 1; s.n0 += 1; cc[sym] += 1; s.n[ctx] = n + 1
        return -L2(p)

def ibits(v):  # cost of the raw mantissa below the bucket
    return max(0, abs(v).bit_length() - 1)
def bucket(v):
    return (0 if v == 0 else (1 if v > 0 else -1) * abs(v).bit_length())

class IntM:
    """Integer: bucket via Cat with context, raw mantissa bits."""
    def __init__(s): s.m = Cat()
    def cost(s, ctx, v):
        if getattr(s, 'name', None): s.lastkey = (s.name, CUR[0], WI[0]); LOG[s.lastkey].append(v)
        return s.m.cost(ctx, bucket(v), 8) + ibits(v)



TOK = re.compile(r'blk_-?\d+|\d+\.\d+\.\d+\.\d+(?::\d+)?|\d+')
def parse(path):
    """Lines -> (templates, [(unix time, thread, template id, ((type, value), ...))])."""
    tpls = {}; rows = []
    for line in open(path, encoding='latin1'):
        d, t, th, rest = line.rstrip('\r\n').split(' ', 3)
        ts = calendar.timegm(time.strptime(d + t, '%y%m%d%H%M%S'))
        slots = []
        def rep(m):
            s = m.group(0)
            if s.startswith('blk_'): slots.append(('B', int(s[4:]))); return '\x01'
            if '.' in s:
                if ':' in s:
                    ip, port = s.split(':'); slots.extend([('I', ip), ('P', int(port))]); return '\x02:\x03'
                slots.append(('I', s)); return '\x02'
            slots.append(('N', s)); return '\x04'
        tp = TOK.sub(rep, rest)
        rows.append((ts, int(th), tpls.setdefault(tp, len(tpls)), tuple(slots)))
    return list(tpls), rows

LOG = defaultdict(list); NEWC = defaultdict(dict); CUR = [0]
def drop(m):
    if getattr(m, 'name', None): LOG[m.lastkey].pop()
FIXED = [0.0]; WI = [0]
def FIX(b): FIXED[0] += b; return b
tpls, rows = parse(sys.argv[1])
TB = defaultdict(float); TN = defaultdict(int)
TT=defaultdict(float)
def add(k, b):
    TB[k] += b; TN[k] += 1; TT[(k, CUR[0])] += b
COMP = [t.split(':')[0].split(' ')[-1] for t in tpls]
ALLOC_T = {i for i, t in enumerate(tpls) if 'allocateBlock' in t}
NN_T = {i for i, t in enumerate(tpls) if 'FSNamesystem' in t}
def explicit_node(tid, slots):
    # datanode lines that start with "IP:50010"
    t = tpls[tid]
    if t.split(': ', 1)[1].startswith('\x02:\x03') if ': ' in t else False:
        return slots[0][1]
    return None

def infer_nodes(g):
    """Encoder-side: logging node per line index (None if unknown)."""
    import itertools
    nodes = [None] * len(g)
    prq = defaultdict(list)
    for i, (ts, th, tid, sl) in enumerate(g):
        if tid == 2: prq[th].append((int(sl[0][1]), i))
    rec = {}
    for i, (ts, th, tid, sl) in enumerate(g):
        if tid == 3 and prq.get(th):
            N, j = prq[th].pop(0); rec[N] = (j, i, th, sl[2][1], ts)
    stored = [sl[0][1] for ts, th, tid, sl in g if tid == 4]
    if rec:
        R = max(rec) + 1
        dn = {}
        for N, r in rec.items():
            if R - N - 1 >= 1: dn[R - N - 1] = r[3]
        rest = [s for s in stored if s not in dn.values()]
        if R not in dn and rest: dn[R] = rest[0]
        for N, (j, i, th, U, ts) in rec.items():
            nodes[j] = nodes[i] = dn.get(R - N)
        recv = [k for k, r in enumerate(g) if r[2] == 0]
        ks = sorted(dn)
        if recv and len(recv) <= 6:
            best = None
            for perm in itertools.permutations(ks, min(len(recv), len(ks))):
                cost = 0; prev_t = None
                for k, d in zip(recv, perm):
                    ts, th, tid, sl = g[k]; U = sl[1][1]
                    exp_up = rec[R - 1][3] if d == 1 and (R - 1) in rec else dn.get(d - 1)
                    cost += 0 if U == exp_up else 1000
                    pr = rec.get(R - d)
                    if pr: cost += min(20, abs(pr[2] - th - 1))
                for (k1, d1), (k2, d2) in itertools.combinations(zip(recv, perm), 2):
                    if g[k1][0] != g[k2][0] and (g[k1][0] < g[k2][0]) != (d1 < d2): cost += 30
                if best is None or cost < best[0]: best = (cost, perm)
            for k, d in zip(recv, best[1]):
                if best[0] < 1000: nodes[k] = dn[d]
    for k, (ts, th, tid, sl) in enumerate(g):
        if tid in NN_T: nodes[k] = 'NN'
        else:
            e = explicit_node(tid, sl)
            if e: nodes[k] = e
    return nodes

DEL_T = {i for i, t in enumerate(tpls) if t.startswith('INFO dfs.FSDataset: Deleting block')}
INV_T = {i for i, t in enumerate(tpls) if 'invalidSet' in t}
tplM = Cat(); timeM = IntM(); bt0 = IntM(); nodeM = Cat(); thrM = IntM(); thrCat = Cat()
copyM = Cat(); numD = Cat(); SEL3 = defaultdict(lambda: (0.0, 0.0)); snM = Cat(); THRT = {}; rtM = Cat(); RT = defaultdict(list); SEL2 = defaultdict(lambda: (0.0, 0.0)); thrM2 = IntM(); SEL = defaultdict(lambda: (0.0, 0.0)); nhM = Cat(); NH = defaultdict(list); ipM = Cat(); portM = IntM(); numM = Cat(); blkM = IntM()
nodehist = defaultdict(list)  # node -> sorted [(t, thread)]
RANGE = 28233
offset = {}  # (src, dst) -> offset
ctr = defaultdict(list)  # src host -> sorted [(t, counter)]
prev_state = None; prev_t0 = None; END = -1
SL = bool(os.environ.get('SL')); LASTK = {}; PREVP = {}
def port_pred(src, dst, ts):
    if SL:
        e = PREVP.get((src, dst))
        return ((e[1] - 32768) % RANGE, (e[0], 0)) if e else (None, None)
    h = ctr[src]
    i = bisect.bisect_right(h, (ts, 1 << 30))
    ref = h[i - 1] if i else (h[0] if h else None)
    if dst is None or (src, dst) not in offset or ref is None: return None, ref
    return (offset[(src, dst)] + ref[1]) % RANGE, ref
def pres(v, pred): return (v - 32768 - pred + RANGE // 2) % RANGE - RANGE // 2
stats = defaultdict(int)
def gapb(x): return min(6, abs(x).bit_length())
dT = IntM(); dN = Cat(); dK = IntM(); dR = IntM(); dTh = Cat(); dSub = IntM(); dTp = Cat()
DB = defaultdict(float)
def dadd(k, b): DB[k] += b
RESID = bool(os.environ.get('RESID')); INTM = set()
for _n, _m in list(globals().items()):
    if not RESID: break
    if isinstance(_m, IntM): INTM.add(_n); _m.m.name = None
    if isinstance(_m, (Cat, IntM)): _m.name = _n
WIN = int(os.environ.get("WIN", 0)) or len(rows)
allrows = rows
for w0 in range(0, len(allrows), WIN):
    rows = allrows[w0:w0 + WIN]; WI[0] = w0
    nodehist.clear(); offset.clear(); ctr.clear(); THRT.clear(); RT.clear(); NH.clear(); prev_state = None; prev_t0 = None; subh = defaultdict(list)
    groups = {}; DELROWS = []
    for r in rows:
        k = next((v for t, v in r[3] if t == 'B'), None)
        if r[2] in DEL_T: DELROWS.append(r); continue
        groups.setdefault(k, []).append(r)
    assert None not in groups
    _o = None; _idx = {}
    for k in groups:
        st = state_of(k)
        if st is None: continue
        if _o is None: _o = st
        _idx[k] = dist(_o, st)
    groups = dict(sorted(groups.items(), key=lambda kv: (0, _idx[kv[0]]) if kv[0] in _idx else (1, 0)))
    for key, g in groups.items():
        st = state_of(key)
        if st is not None and prev_state is not None:
            add('blkid', blkM.cost('d', dist(prev_state, st)) + 1)
        else: add('blkid', FIX(65))
        if st is not None: prev_state = st
        ALLOC = ALLOC_T
        g = sorted(g, key=lambda r: (r[0], 0 if r[2] == 0 else (1 if r[2] in ALLOC else 2)))
        t0 = g[0][0]
        add('time', bt0.cost('first', t0 - prev_t0 if prev_t0 is not None else 0)); prev_t0 = t0
        nodes = infer_nodes(g)
        dn_all = [n for n in nodes if n not in (None, 'NN')]
        hist = []; vals = defaultdict(list); pt = t0; dprev = 0
        vals['B'] = [key]; seen_tn = []; samenode = {}; upst = {}
        lastthr = {}  # node -> (thread, tid) within block
        for (ts, th, tid, slots), node in zip(g, nodes):
            CUR[0] = tid
            add('tpl', tplM.cost(tuple(hist[-2:]), tid, 8))
            # hidden node
            if tid == 0:
                src, v = slots[1][1], slots[2][1]
                cands = [node] + [x for x in vals['I'] + list(dict.fromkeys(dn_all)) if x != node]
                sc = []
                for c in cands:
                    p, _ = port_pred(src, c, ts)
                    if p is not None: sc.append((abs(pres(v, p)), c != node, c))
                if sc and (min(sc)[0] < 64) and min(sc)[2] != node:
                    node = min(sc)[2]; stats['reassign'] += 1
            if node is None and tid not in NN_T and tid not in (0, 2, 3) and vals['I']:
                sub = next((v for ty, v in slots if ty == 'N'), None)
                best = None
                usedn = {n for (t_, n) in seen_tn if t_ == tid}
                for x in vals['I']:
                    if x in usedn: continue
                    sc = (2 if ts in RT[(tid, x)] else 0) + (3 if sub is not None and NH[(tid, 1, x)][:1] == [sub] else 0)
                    if best is None or sc > best[0]: best = (sc, x)
                if best: node = best[1]
            if node == 'NN': pass
            else:
                lst = vals['I']
                sym = 'unk' if node is None else (lst.index(node) if node in lst else 'new')
                b = nodeM.cost((tid, min(len(lst), 4)), sym, 4)
                if sym == 'new': b += ipM.cost('node', node, 32); lst.insert(0, node)
                elif sym != 'unk': lst.insert(0, lst.pop(sym))
                add('node', b); stats['unk' if node is None else 'known'] += 1
            if hist:
                d = ts - pt
                ca = timeM.cost((tid, hist[-1], gapb(dprev)), d)
                rl = RT[(tid, node)]
                sym = rl.index(ts) if ts in rl else 'miss'
                cb = rtM.cost((tid, min(len(rl), 4)), sym, 5) + (ca if sym == 'miss' else 0)
                key2 = (tid, hist[-1]); sa, sb = SEL2[key2]
                (drop(rtM) if sa <= sb else (drop(timeM) if sym != 'miss' else None))
                add('time', ca if sa <= sb else cb); SEL2[key2] = (sa * 0.98 + ca, sb * 0.98 + cb)
                dprev = d
            else:
                rl = RT[(tid, node)]
            if ts in rl: rl.remove(ts)
            rl.insert(0, ts); del rl[16:]
            pt = ts
            # thread
            if (node == 'NN' or node is None) and (tid, ts) in THRT:
                hit = THRT[(tid, ts)] == th
                add('thread', thrCat.cost(('rt', tid), hit, 2))
            if (node == 'NN' or node is None) and (tid, ts) in THRT and THRT[(tid, ts)] == th: pass
            elif node == 'NN' or node is None:
                if th < 100: add('thread', thrCat.cost(('s', tid), th, 10))
                else:
                    add('thread', thrCat.cost(('s', tid), 'big', 10) + thrM.cost(('big', tid), th - 200))
            else:
                h = nodehist[(node, COMP[tid])] if not SL else LASTK.get((node, COMP[tid]), [])
                i = bisect.bisect_left(h, (ts, -1)); j = bisect.bisect_right(h, (ts, 1 << 30))
                lo = h[i - 1][1] if i else None
                same = [x[1] for x in h[i:j]]
                hi = h[j][1] if j < len(h) else None
                if same: base = same[0]; c = ('same', tid, min(8, len(same)))
                elif lo is not None and hi is not None: base = lo; c = ('br', tid, gapb(hi - lo))
                elif lo is not None: base = lo; c = ('lo', tid, gapb(ts - h[i - 1][0]))
                elif hi is not None: base = hi; c = ('hi', tid, gapb(h[j][0] - ts))
                else: base = 200; c = ('none', tid)
                cb = thrM.cost(c, th - base)
                if node in lastthr:
                    pth, ptid, pts = lastthr[node]
                    key = (tid, ptid)
                    ca = thrM2.cost((tid, ptid, gapb(ts - pts)), th - pth)
                    sa, sb = SEL[key]
                    drop(thrM) if sa < sb else drop(thrM2)
                    add('thread', (ca if sa < sb else cb) + 0.0)
                    SEL[key] = (sa * 0.98 + ca, sb * 0.98 + cb)
                else:
                    add('thread', cb)
            if node not in (None, 'NN'):
                lastthr[node] = (th, tid, ts); bisect.insort(nodehist[(node, COMP[tid])], (ts, th)); LASTK[(node, COMP[tid])] = [(ts, th)]
            THRT[(tid, ts)] = th; seen_tn.append((tid, node))
            hist.append(tid)
            first_b = True; previp = None
            _rec = node not in (None, 'NN')
            if tid == 0 and node not in (None, 'NN'): upst[node] = slots[1][1]
            role = -1
            if node not in (None, 'NN'):
                role, cur, vis = 0, node, {node}
                while True:
                    nx = next((y for y, u in upst.items() if u == cur and y not in vis), None)
                    if nx is None: break
                    role += 1; cur = nx; vis.add(nx)
            for si, (ty, v) in enumerate(slots):
                sctx = (tid, si)
                if ty == 'B' and first_b: first_b = False; continue
                if ty == 'P':
                    if v == 50010: add('port', copyM.cost(sctx, '50010', 4)); continue
                    b = copyM.cost(sctx, 'p', 4)
                    src, dst = previp, node
                    h = ctr[src]
                    pred, ref = port_pred(src, dst, ts)
                    if pred is not None:
                        d = pres(v, pred)
                        b += copyM.cost(('pk', gapb(ts - ref[0])), 'k', 4) + portM.cost(('k', gapb(ts - ref[0])), d)
                        if SL: c = 0
                        elif abs(d) < 64: c = (v - 32768 - offset[(src, dst)]) % RANGE
                        else: c = ref[1]; offset[(src, dst)] = (v - 32768 - c) % RANGE
                        stats['port_pred'] += 1
                    else:
                        b += copyM.cost(('pk', 0), 'raw', 4) + FIX(14.8)
                        c = ref[1] if ref is not None else 0
                        if dst is not None: offset[(src, dst)] = (v - 32768 - c) % RANGE
                        stats['port_raw'] += 1
                    bisect.insort(h, (ts, c)); PREVP[(src, dst)] = (ts, v)
                    add('port', b); continue
                lst = vals[ty]
                ref = samenode.get((node, ty)) if node not in (None, 'NN') else None
                if ref is not None:
                    rt_, rv = ref
                    sym = rv.index(v) if v in rv else 'no'
                    bsn = snM.cost((tid, si, rt_), sym, 3)
                    if sym != 'no':
                        add('sn' + ty, bsn)
                        if v in lst: lst.remove(v)
                        lst.insert(0, v)
                        if ty == 'I': previp = v
                        continue
                    add('sn' + ty, bsn)
                if v in lst:
                    i = lst.index(v); b = copyM.cost((sctx, min(len(lst), 4)), i, 4); lst.insert(0, lst.pop(i))
                    add('copy' + ty, b)
                else:
                    b = copyM.cost((sctx, min(len(lst), 4)), 'new', 4)
                    if ty == 'I': b += ipM.cost(sctx, v, 32)
                    elif ty == 'N':
                        b0 = b
                        kip = node if node not in (None, 'NN') else (vals['I'][-1] if vals['I'] else None)
                        # same int as an earlier N slot in this line
                        eq = [j for j in range(si) if slots[j][0] == 'N' and int(slots[j][1]) == int(v)]
                        l1 = NH[(tid, si, kip)]; l2 = NH[(tid, si)]
                        if eq: sym = ('eq', eq[-1])
                        elif v in l1: sym = ('h1', l1.index(v))
                        elif v in l2: sym = ('h2', min(l2.index(v), 64))
                        else: sym = 'new'
                        b += nhM.cost((tid, si, role), sym if sym[0] != 'h2' else ('h2', l2.index(v).bit_length()), 6)
                        if sym != 'new' and sym[0] == 'h2': b += max(0, l2.index(v).bit_length() - 1)
                        if sym == 'new': b += numM.cost((tid, si, role), v, 4 * len(v))
                        bd = numD.cost((tid, si, role, hist[-2] if len(hist) > 1 else None), v, 4 * len(v) + 2)
                        sa_, sb_ = SEL3[(tid, si)]
                        (drop(numD) if sa_ <= sb_ else (drop(nhM), drop(numM) if sym == 'new' else None))
                        b = b0 + (b - b0 if sa_ <= sb_ else bd); SEL3[(tid, si)] = (sa_ * 0.98 + (b - b0), sb_ * 0.98 + bd) if sa_ <= sb_ else (sa_ * 0.98, sb_ * 0.98 + bd)
                        for l, cap in ((l1, 8), (l2, 256)):
                            if v in l: l.remove(v)
                            l.insert(0, v); del l[cap:]
                    else: b += FIX(64)
                    add('new' + ty, b); lst.insert(0, v)
                if ty == 'I': previp = v
            if _rec:
                for ty_ in ('I', 'N'):
                    vv = [v for t2, v in slots if t2 == ty_]
                    if vv: samenode[(node, ty_)] = (tid, vv)
        add('tpl', tplM.cost(tuple(hist[-2:]), END, 8))
    pend = defaultdict(list); pipe = defaultdict(set)
    ev = []
    for r in rows:
        if r[2] in INV_T: ev.append((r[0], 0, r))
    for r in DELROWS: ev.append((r[0], 1, r))
    ev.sort(key=lambda e: (e[0], e[1]))
    # encoder: assign nodes
    assigned = []
    for ts, kind, r in ev:
        if kind == 0:
            b = r[3][0][1]; X = r[3][1][1]; pipe[b].add(X); bisect.insort(pend[X], b)
        else:
            b = r[3][0][1]; best = None
            for X in sorted(pipe.get(b, ())):
                p = pend[X]; i = bisect.bisect_left(p, b)
                if i < len(p) and p[i] == b and (best is None or i < best[0]): best = (i, X)
            if best is None: assigned.append((r, None, None)); continue
            i, X = best; pend[X].pop(i); pipe[b].discard(X); assigned.append((r, X, i))
    # coding: per second, per node (sorted), blocks ascending; ranks over replayed pending sets
    pend2 = defaultdict(list)
    invs = sorted(((r[0], r[3][0][1], r[3][1][1]) for r in rows if r[2] in INV_T))
    grp = defaultdict(list); esc = []
    for r, X, i in assigned:
        if X is None: esc.append(r)
        else: grp[(r[0], X)].append(r)
    mtf = []; lastk = {}; pts = None; ii = 0; ptid = None
    for (ts, X) in sorted(grp):
        while ii < len(invs) and invs[ii][0] <= ts:
            bisect.insort(pend2[invs[ii][2]], invs[ii][1]); ii += 1
        dadd('del_time', dT.cost('d', ts - pts if pts is not None else 0)); 
        sym = mtf.index(X) if X in mtf else 'new'
        dadd('del_node', dN.cost(ts == pts, sym, 8)); 
        if X in mtf: mtf.remove(X)
        mtf.insert(0, X); del mtf[64:]
        pts = ts
        rs = sorted(grp[(ts, X)], key=lambda r: r[3][0][1])
        dadd('del_count', dK.cost(min(6, lastk.get(X, 0).bit_length()), len(rs) - 1)); lastk[X] = len(rs)
        pr = -1
        for r in rs:
            _, th, tid, sl = r; bb = sl[0][1]
            p = pend2[X]; i = bisect.bisect_left(p, bb)
            if i < len(p) and p[i] == bb: p.pop(i); dadd('del_rank', dR.cost(min(pr, 4), i)); pr = i
            else: dadd('del_rank', dR.cost('miss', 0) + FIX(40))  # rare: assigned but already removed
            dadd('del_thread', dTh.cost(X, th, 6))
            dadd('del_tpl', dTp.cost(ptid, tid, 2)); ptid = tid
            sub = next((int(v) for t, v in sl if t == 'N'), None)
            if sub is not None:
                st = state_of(bb); ix = dist(_o, st) if st is not None and _o is not None else 0
                h = subh[X]; j = bisect.bisect_left(h, (ix, -1))
                nb = [h[k] for k in (j - 1, j) if 0 <= k < len(h)]
                if nb:
                    near = min(nb, key=lambda e: abs(e[0] - ix)); c = ('nb', min(12, abs(near[0] - ix).bit_length()))
                    dadd('del_sub', dSub.cost(c, sub - near[1]))
                else: dadd('del_sub', dSub.cost('none', sub))
                bisect.insort(h, (ix, sub))
    # deletes with no pending entry in this window: time, LCG index of the block, subdir
    pe = None
    for r in sorted(esc, key=lambda r: (r[0], r[3][0][1])):
        ts, th, tid, sl = r; bb = sl[0][1]; st = state_of(bb)
        dadd('del_esc', dT.cost('esc', ts - pe if pe is not None else 0) + dTh.cost('esc', th, 6) + dTp.cost('esc', tid, 2))
        dadd('del_esc', dK.cost('escb', dist(_o, st)) if st is not None and _o is not None else FIX(65))
        sub = next((int(v) for t, v in sl if t == 'N'), None)
        if sub is not None: dadd('del_esc', dSub.cost('escsub', sub))
        pe = ts
for k in sorted(TB, key=lambda k: -TB[k]):
    print(f'{k:10s} {TB[k]/8:12,.0f} B  {TN[k]:9d} n  {TB[k]/max(1,TN[k]):6.2f} b/ea')
print(f'total {sum(TB.values())/8:,.0f} B', dict(stats))

for k in ('thread','time','node','newN','tpl','newI','copyI'):
    top=sorted(((v,t) for (kk,t),v in TT.items() if kk==k), reverse=True)[:5]
    print(k, '  '.join(f'{t}:{v/8/1000:.0f}K' for v,t in top))

for k in sorted(DB, key=lambda k: -DB[k]): print(f'{k:10s} {DB[k]/8:12,.0f} B')
print(f'delete stream {sum(DB.values())/8:,.0f} B ; grand total {(sum(DB.values()) + sum(TB.values()))/8:,.0f} B')

if RESID:  # residual streams through raw LZMA2, one stream per model and template (and window)
    import lzma
    def varint(z, out):
        while True:
            b = z & 127; z >>= 7
            if z: out.append(b | 128)
            else: out.append(b); return
    tot_h0 = tot_xz = 0; per = defaultdict(lambda: [0.0, 0])
    for key, syms in LOG.items():
        if not syms: continue
        ints = all(isinstance(x, int) and not isinstance(x, bool) for x in syms) and key[0] in INTM
        if ints:  # bucket order-0 + raw mantissa, and zigzag varints through xz
            from collections import Counter as C_
            bc = C_(bucket(v) for v in syms); n = len(syms)
            h = -sum(c * math.log2(c / n) for c in bc.values()) + sum(ibits(v) for v in syms)
            buf = bytearray()
            for v in syms: varint((v << 1) ^ (v >> 63) if v >= 0 else ((-v) << 1) - 1, buf)
        else:  # symbol ids by first appearance; literals priced at their escape cost
            from collections import Counter as C_
            c = C_(map(repr, syms)); n = len(syms)
            lit = sum(NEWC[key].get(x, 8) for x in set(syms) if not isinstance(x, int) or key[0] in ('ipM', 'numM', 'numD', 'thrCat', 'dTh', 'dN'))
            h = -sum(v * math.log2(v / n) for v in c.values()) + lit
            ids = {}; buf = bytearray()
            for x in syms: varint(ids.setdefault(repr(x), len(ids)), buf)
            h_xz_lit = lit
        xz = len(lzma.compress(bytes(buf), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}])) * 8 + (0 if ints else lit)
        tot_h0 += h; tot_xz += xz; per[key[0]][0] += h; per[key[0]][1] += xz
    print(f'fixed literals {FIXED[0]/8:,.0f} B'); tot_h0 += FIXED[0]; tot_xz += FIXED[0]
    print(len([1 for v in LOG.values() if v]), 'streams'); print(f'residual streams: order-0 {tot_h0/8:,.0f} B, xz per stream {tot_xz/8:,.0f} B')
    for k, (h, x) in sorted(per.items(), key=lambda kv: -kv[1][1])[:14]: print(f'  {k:8s} order0 {h/8:10,.0f}  xz {x/8:10,.0f}')
