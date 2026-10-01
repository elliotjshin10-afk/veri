/* Client-side address scoring.
 *
 * TronGrid serves cross-origin requests, so the page fetches an address's
 * TRC-20 history itself and scores it here. No backend, and it works on any
 * Tron address rather than only the ones we pre-computed.
 *
 * Everything below must match src/veridis/features/asof.py exactly. Where the
 * two drift, the model is being fed something it was not trained on - the
 * classic train/serve skew - so scripts/check_parity.py re-derives these same
 * features in Python for a sample of real addresses and fails if any score
 * differs. Change one side, run that, or this file is lying.
 */
/* TronGrid access.
 *
 * The key is sent from the browser by design. It is a limited key and the
 * owner has accepted that it is public here; anyone reading this file can see
 * it. Without it the keyless tier throttles after a couple of lookups, which
 * a visitor notices immediately.
 *
 * If it ever needs to be private, deploy proxy/ (a Cloudflare Worker holding
 * the key as a secret), call setProxy() with its URL, and clear API_KEY - the
 * request path below already supports that with no other change. */
export const API_KEY = "dbbdc932-2d89-426f-8537-85bfbcec2c16";
export let PROXY_BASE = "";
export function setProxy(base){ PROXY_BASE = (base || "").replace(/\/$/, ""); }
const apiBase = () => PROXY_BASE || "https://api.trongrid.io";
// A proxy supplies the key itself, so never send it there too.
const authHeaders = () =>
  (!PROXY_BASE && API_KEY) ? {"TRON-PRO-API-KEY": API_KEY} : {};

const DAY_MS = 86400000, EPS = 1e-9;
const STABLES = new Set(["USDT","USDC","TUSD","USDD","DAI","FDUSD","PYUSD"]);

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

export async function fetchTransfers(address, {maxPages = 5, signal, onProgress} = {}) {
  const out = [];
  let hitCap = true;   // cleared when a page comes back short or a cursor runs out
  let url = `${apiBase()}/v1/accounts/${address}/transactions/trc20`
          + `?limit=200&order_by=block_timestamp,desc`;
  for (let page = 0; page < maxPages; page++) {
    // TronGrid's keyless tier throttles, and it throttles per IP - which here
    // is the visitor's own, not a shared server's, so one person's browsing
    // does not spend anybody else's budget. A short backoff covers the bursts
    // that a few lookups in a row produce; sustained traffic needs an API key.
    let res = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      if (onProgress && attempt) onProgress("retrying");
      res = await fetch(url, {signal, headers: authHeaders()});
      if (res.status !== 429) break;
      await sleep(900 * Math.pow(2, attempt));
    }
    if (!res || res.status === 429) throw new Error("rate_limited");
    if (!res.ok) throw new Error("fetch_failed");
    const j = await res.json();
    for (const r of (j.data || [])) {
      if (r.type !== "Transfer") continue;
      const info = r.token_info || {};
      const sym = (info.symbol || "").toUpperCase();
      if (!STABLES.has(sym)) continue;
      const dec = Number(info.decimals ?? 6);
      const v = Number(r.value) / Math.pow(10, dec);
      if (!isFinite(v) || v <= 0) continue;
      out.push({from: r.from, to: r.to, usd: v, t: Number(r.block_timestamp)});
    }
    let next = j.meta && j.meta.links && j.meta.links.next;
    if (!next || !(j.data || []).length) { hitCap = false; break; }
    // Pagination links come back absolute, pointing at api.trongrid.io. Left
    // as-is they would bypass the proxy from page two onward and silently drop
    // back to the keyless limit.
    if (PROXY_BASE) next = next.replace(/^https:\/\/api\.trongrid\.io/, PROXY_BASE);
    url = next;
  }
  // Metadata rides on the array so every existing caller keeps working: the
  // return value is still exactly the list of transfers.
  Object.defineProperty(out, "truncated", {value: hitCap, enumerable: false});
  return out;
}

/* The genuine first stablecoin transfer, in one request. Only worth asking for
   when the main fetch hit its page cap - otherwise the window already contains
   the whole history and its oldest row IS the first transfer. Failure is not an
   error: we fall back to the window's own earliest row, which is what the code
   did before this existed. */
export async function fetchFirstSeen(address, {signal} = {}) {
  const url = `${apiBase()}/v1/accounts/${address}/transactions/trc20`
            + `?limit=1&order_by=block_timestamp,asc`;
  try {
    const res = await fetch(url, {signal, headers: authHeaders()});
    if (!res.ok) return null;
    const j = await res.json();
    const r = (j.data || [])[0];
    return r ? Number(r.block_timestamp) : null;
  } catch { return null; }
}

/* DuckDB's MEDIAN / QUANTILE_CONT use linear interpolation. */
function quantile(sorted, q) {
  if (!sorted.length) return null;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/* The fills `_derive` applies when an address has no legs before the cutoff. */
function EMPTY_FEATURES() {
  return {
    dest_age_days: 0, dest_inbound_count: 0, dest_outbound_count: 0,
    dest_senders_all: 0, dest_senders_1d: 0, dest_senders_7d: 0,
    dest_senders_30d: 0, dest_inbound_usd_7d: 0, dest_inbound_usd_30d: 0,
    dest_inbound_count_7d: 0, dest_distinct_out: 0,
    dest_consolidation_ratio: 0, dest_median_hold_secs: -1,
    dest_p25_hold_secs: -1, dest_in_out_ratio: 0,
    // These two stay null rather than 0. `_derive` fills most columns but not
    // these, so Python hands LightGBM a missing value and the tree takes its
    // default branch. A zero would take a real branch and score differently.
    dest_forward_ratio: null, dest_usd_per_sender: null,
  };
}

/* `opts.firstSeen` overrides the earliest timestamp when the caller knows the
   fetched window is not the whole history. TronGrid pages newest-first, so an
   address busier than the page budget yields a window whose oldest row is
   recent - and age, computed from that, comes out days old for a wallet that
   has run for years. That inflates risk on exactly the busiest addresses.
   The fix is one extra ascending request for the genuine first transfer. */
export function computeFeatures(transfers, address, asOfMs, opts = {}) {
  // Long form, mirroring the `legs` table: one row per (address, side).
  const legs = [];
  for (const r of transfers) {
    if (r.t >= asOfMs) continue;               // the leakage guard
    // Zero-value transfers are dropped in the SQL the model was trained on
    // (`WHERE amount_usd > 0`). Filtering only at fetch time left this path
    // able to see them, which moved counts, hold times and the score.
    if (!(r.usd > 0)) continue;
    if (r.to === address)   legs.push({side: "in",  peer: r.from, usd: r.usd, t: r.t});
    if (r.from === address) legs.push({side: "out", peer: r.to,   usd: r.usd, t: r.t});
  }
  // An address whose transfers are all zero-value (or all after the cutoff)
  // still produces a feature row in Python - the SQL join yields nothing and
  // `_derive` fills the defaults - so returning null here would score two
  // addresses differently for no reason. The page decides separately whether
  // it is worth showing a result at all.
  if (!legs.length) return EMPTY_FEATURES();
  // Ties ordered explicitly, matching `ORDER BY block_time, side` in the SQL:
  // 'in' before 'out', so an arrival and a sweep in the same block reads as a
  // zero-second hold rather than depending on sort order.
  legs.sort((a, b) => a.t - b.t || (a.side < b.side ? -1 : a.side > b.side ? 1 : 0));

  const inLegs  = legs.filter(l => l.side === "in");
  const outLegs = legs.filter(l => l.side === "out");
  const since = (ms) => asOfMs - ms;
  const distinct = (rows) => new Set(rows.map(r => r.peer)).size;
  const sum = (rows) => rows.reduce((a, r) => a + r.usd, 0);

  const inboundUsd = sum(inLegs), outboundUsd = sum(outLegs);
  const sendersAll = distinct(inLegs);
  // A caller-supplied first-seen only ever moves age EARLIER (a longer life),
  // so a stale or missing probe can never inflate the score.
  const firstSeen = (opts.firstSeen != null && opts.firstSeen < legs[0].t)
    ? opts.firstSeen : legs[0].t;

  // Consolidation: share of outbound value to the single most used payee.
  const byPeer = new Map();
  for (const l of outLegs) byPeer.set(l.peer, (byPeer.get(l.peer) || 0) + l.usd);
  const outTotal = [...byPeer.values()].reduce((a, b) => a + b, 0);
  const consolidation = outTotal > 0 ? Math.max(...byPeer.values()) / outTotal : null;

  // Causal sweep gap: for each outbound, seconds since the most recent inbound
  // strictly before it. Looks only backward, exactly like the SQL window.
  const holds = [];
  let lastIn = null;
  for (const l of legs) {
    if (l.side === "in") lastIn = l.t;
    else if (lastIn !== null) holds.push((l.t - lastIn) / 1000);
  }
  holds.sort((a, b) => a - b);

  const win = (rows, days) => rows.filter(r => since(r.t) <= days * DAY_MS);

  return {
    dest_age_days: (asOfMs - firstSeen) / DAY_MS,
    dest_inbound_count: inLegs.length,
    dest_outbound_count: outLegs.length,
    dest_senders_all: sendersAll,
    dest_senders_1d: distinct(win(inLegs, 1)),
    dest_senders_7d: distinct(win(inLegs, 7)),
    dest_senders_30d: distinct(win(inLegs, 30)),
    dest_inbound_usd_7d: sum(win(inLegs, 7)),
    dest_inbound_usd_30d: sum(win(inLegs, 30)),
    dest_inbound_count_7d: win(inLegs, 7).length,
    dest_distinct_out: byPeer.size,
    dest_consolidation_ratio: consolidation === null ? 0 : consolidation,
    dest_median_hold_secs: holds.length ? quantile(holds, 0.5) : -1,
    dest_p25_hold_secs: holds.length ? quantile(holds, 0.25) : -1,
    dest_forward_ratio: outboundUsd / (inboundUsd + EPS),
    dest_in_out_ratio: inLegs.length / (outLegs.length + EPS),
    dest_usd_per_sender: inboundUsd / (sendersAll + EPS),
  };
}

function walk(node, x) {
  while (node.v === undefined) {
    const val = x[node.f];
    // LightGBM sends missing values the default direction; ours are never NaN,
    // but mirror the behaviour rather than assume it.
    node = (val === null || val === undefined || Number.isNaN(val))
      ? (node.d ? node.l : node.r)
      : (val <= node.t ? node.l : node.r);
  }
  return node.v;
}

export function score(model, features) {
  const x = model.features.map(f => features[f]);
  let raw = 0;
  for (const t of model.trees) raw += walk(t, x);
  return 1 / (1 + Math.exp(-raw));       // LightGBM binary objective
}

export function band(p, thresholds) {
  if (p >= thresholds.high) return "high";
  if (p >= thresholds.elevated) return "elevated";
  return "ordinary";
}

/* ── the two-sided check ────────────────────────────────────────────────────
   An address lookup can only judge the destination. A real send also has a
   sender and an amount, and that is a materially better model: on the same
   event-level test set, 0.936 ROC-AUC and 24.2% recall at a 1% false-alarm
   budget become 0.974 and 37.3%.

   Every feature below comes from the two addresses' own histories. Nothing
   here needs a third fetch - `shared_counterparties` is the intersection of
   two counterparty sets we already hold.

   Each block names the SQL in src/veridis/features/asof.py it mirrors.
   scripts/check_pair_parity.py runs both over real sends and requires exact
   agreement, so anything that drifts here fails the build rather than quietly
   scoring people wrong. */

const HOUR_MS = 3600000;

/* Build the same `legs` long form the SQL does: one row per (address, side),
   zero-value transfers dropped (`WHERE amount_usd > 0`), strictly before the
   event. `peer` is the counterparty on the other end of that leg. */
function legsFor(transfers, address, asOfMs) {
  const legs = [];
  for (const r of transfers) {
    if (r.t >= asOfMs) continue;
    if (!(r.usd > 0)) continue;
    if (r.to === address) legs.push({side: "in", peer: r.from, usd: r.usd, t: r.t});
    if (r.from === address) legs.push({side: "out", peer: r.to, usd: r.usd, t: r.t});
  }
  legs.sort((a, b) => a.t - b.t || (a.side < b.side ? -1 : a.side > b.side ? 1 : 0));
  return legs;
}

/* Context + relationship features, given both sides and the amount.
   `destFeatures` is the output of computeFeatures() for the destination. */
export function pairFeatures(opts) {
  const {senderTransfers, destTransfers, sender, destination,
         amountUsd, asOfMs, destFeatures} = opts;
  const EPS2 = 1e-9;
  const sLegs = legsFor(senderTransfers, sender, asOfMs);
  const dLegs = legsFor(destTransfers, destination, asOfMs);
  const sOut = sLegs.filter(l => l.side === "out");

  /* _PAIR_SQL: the sender's prior outbound legs to THIS destination. */
  const pair = sOut.filter(l => l.peer === destination);
  const n = pair.length;
  const pairPriorUsd = pair.reduce((a, l) => a + l.usd, 0);
  const pairMax = n ? Math.max(...pair.map(l => l.usd)) : null;
  // ARG_MIN(amount_usd, block_time): the amount of the EARLIEST prior send,
  // not the smallest one. Ties broken by order, as the sort above fixes.
  const pairFirst = n ? pair.reduce((b, l) => (l.t < b.t ? l : b), pair[0]).usd : null;
  const pairLastT = n ? Math.max(...pair.map(l => l.t)) : null;
  const pairMinT = n ? Math.min(...pair.map(l => l.t)) : null;

  /* _CONCENTRATION_SQL: how much of the sender's outflow this pair carries. */
  const pairOut7d = sOut.filter(l => l.peer === destination && l.t >= asOfMs - 7 * DAY_MS)
                        .reduce((a, l) => a + l.usd, 0);
  const pairOutAll = sOut.filter(l => l.peer === destination).reduce((a, l) => a + l.usd, 0);
  const senderOut7d = sOut.filter(l => l.t >= asOfMs - 7 * DAY_MS).reduce((a, l) => a + l.usd, 0);
  const senderLifetimeOut = sOut.reduce((a, l) => a + l.usd, 0);

  /* _SHARED_SQL: DISTINCT peers on each side, then the intersection. Both
     sides use ALL legs, not just outbound. */
  const sPeers = new Set(sLegs.map(l => l.peer));
  const dPeers = new Set(dLegs.map(l => l.peer));
  let shared = 0;
  for (const p of sPeers) if (dPeers.has(p)) shared++;

  /* Mean hour of the sender's outbound legs, by UTC arithmetic. The SQL used
     EXTRACT(hour FROM to_timestamp(...)) until this check caught it resolving
     in the machine's local timezone.

     A sender with no outbound history gets -1, not null: _derive fills
     sender_mean_out_hour with -1 BEFORE subtracting, so Python yields a real
     number (|event_hour + 1|) where a null would have been. Returning null
     here instead sent those rows down the tree's default branch and scored
     them differently - which is exactly the skew this check exists to find. */
  const meanOutHour = sOut.length
    ? sOut.reduce((a, l) => a + (Math.floor(l.t / HOUR_MS) % 24), 0) / sOut.length
    : -1;
  const eventHour = Math.floor(asOfMs / HOUR_MS) % 24;

  const clip01 = (v) => Math.min(1, Math.max(0, v));
  const destAge = destFeatures && destFeatures.dest_age_days != null
    ? destFeatures.dest_age_days : 0;

  return {
    // context
    amount_usd: amountUsd,
    event_hour: eventHour,
    hour_deviation: Math.abs(eventHour - meanOutHour),
    amount_round_1k: amountUsd % 1000 === 0 ? 1 : 0,
    amount_round_100: amountUsd % 100 === 0 ? 1 : 0,
    // relationship - null-filled exactly as _derive does
    sender_prior_sends_to_dest: n,
    pair_prior_usd: pairPriorUsd,
    pair_secs_since_last: pairLastT === null ? -1 : (asOfMs - pairLastT) / 1000,
    pair_span_days: pairLastT === null ? -1 : (pairLastT - pairMinT) / DAY_MS,
    escalation_ratio: pairMax === null ? -1 : amountUsd / (pairMax + EPS2),
    escalation_vs_first: pairFirst === null ? -1 : amountUsd / (pairFirst + EPS2),
    pair_peak_vs_first: pairFirst === null ? -1 : pairMax / (pairFirst + EPS2),
    is_first_send_to_dest: n === 0 ? 1 : 0,
    first_send_to_young_dest: (n === 0 && destAge < 30) ? 1 : 0,
    shared_counterparties: shared,
    no_shared_counterparties: shared === 0 ? 1 : 0,
    pair_share_of_outflow_7d: clip01(pairOut7d / (senderOut7d + EPS2)),
    pair_share_of_outflow_all: clip01(pairOutAll / (senderLifetimeOut + EPS2)),
    pair_outflow_usd_7d: pairOut7d,
    pair_outflow_usd_all: pairOutAll,
  };
}

/* Everything the two-sided model needs, in one object. */
export function computePairFeatures(opts) {
  const dest = computeFeatures(opts.destTransfers, opts.destination, opts.asOfMs,
                               {firstSeen: opts.destFirstSeen});
  return Object.assign({}, dest, pairFeatures(Object.assign({destFeatures: dest}, opts)));
}

/* ── address validation ─────────────────────────────────────────────────────
   A Tron address is base58check: 0x41, 20 address bytes, then the first four
   bytes of SHA-256(SHA-256(body)). Checking only the alphabet and the length
   accepts a typo, and TronGrid then answers 400 and the page says "could not
   read that address just now" - which blames the network for the user's
   slipped keystroke. On a tool about irreversible payments, "that address has
   a typo" is the single most useful thing we can say.

   SHA-256 inline and synchronous: crypto.subtle is async, and making address
   validation async would push a promise through every call site for 30 lines
   of arithmetic. Verified against Python's hashlib over real addresses. */
const K256 = new Uint32Array([
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);

function sha256(bytes) {
  const H = new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                             0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);
  const len = bytes.length, bitLen = len * 8;
  const padded = new Uint8Array((((len + 9) >> 6) + 1) << 6);
  padded.set(bytes); padded[len] = 0x80;
  new DataView(padded.buffer).setUint32(padded.length - 4, bitLen >>> 0);
  const w = new Uint32Array(64);
  const rot = (x, n) => (x >>> n) | (x << (32 - n));
  for (let i = 0; i < padded.length; i += 64) {
    for (let j = 0; j < 16; j++)
      w[j] = (padded[i+j*4] << 24) | (padded[i+j*4+1] << 16)
           | (padded[i+j*4+2] << 8) | padded[i+j*4+3];
    for (let j = 16; j < 64; j++) {
      const s0 = rot(w[j-15],7) ^ rot(w[j-15],18) ^ (w[j-15] >>> 3);
      const s1 = rot(w[j-2],17) ^ rot(w[j-2],19) ^ (w[j-2] >>> 10);
      w[j] = (w[j-16] + s0 + w[j-7] + s1) >>> 0;
    }
    let [a,b,c,d,e,f,g,h] = H;
    for (let j = 0; j < 64; j++) {
      const S1 = rot(e,6) ^ rot(e,11) ^ rot(e,25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K256[j] + w[j]) >>> 0;
      const S0 = rot(a,2) ^ rot(a,13) ^ rot(a,22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      h=g; g=f; f=e; e=(d+t1)>>>0; d=c; c=b; b=a; a=(t1+t2)>>>0;
    }
    const v=[a,b,c,d,e,f,g,h];
    for (let j = 0; j < 8; j++) H[j] = (H[j] + v[j]) >>> 0;
  }
  const out = new Uint8Array(32);
  for (let j = 0; j < 8; j++) {
    out[j*4]=H[j]>>>24; out[j*4+1]=(H[j]>>>16)&255;
    out[j*4+2]=(H[j]>>>8)&255; out[j*4+3]=H[j]&255;
  }
  return out;
}

const B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/** True only for a base58check-valid Tron mainnet address. */
export function isValidTronAddress(a) {
  if (typeof a !== "string" || a.length !== 34 || a[0] !== "T") return false;
  let n = 0n;
  for (const ch of a) {
    const i = B58_ALPHABET.indexOf(ch);
    if (i < 0) return false;
    n = n * 58n + BigInt(i);
  }
  const bytes = new Uint8Array(25);
  for (let i = 24; i >= 0; i--) { bytes[i] = Number(n & 255n); n >>= 8n; }
  if (n !== 0n || bytes[0] !== 0x41) return false;
  const sum = sha256(sha256(bytes.subarray(0, 21)));
  for (let i = 0; i < 4; i++) if (sum[i] !== bytes[21 + i]) return false;
  return true;
}

/* ── Ethereum ───────────────────────────────────────────────────────────────
   Etherscan rather than Blockscout, for one reason that decides correctness
   and two that are convenience.

   The deciding one: Etherscan sorts ASCENDING. Blockscout pages newest-first
   and refuses sort=asc with a 422, so for an address busier than the page
   budget there is no cheap way to learn when it started - and age, measured
   from the first transfer, is then wrong at every point in time, along with
   every lifetime aggregate built on it. That is the difference between a
   week-old collector and a three-year-old business. With an ascending sort the
   first page IS the beginning, so age is exact for everyone.

   The conveniences: a higher rate limit (Blockscout started refusing during
   our own ingest), and the same key reaches Arbitrum and Polygon on the free
   tier, which is where this goes next.

   Verified against the data the model was trained on - 150 transfers across
   six addresses, zero disagreement, once the feature layer's own
   `amount_usd > 0` filter is applied to both sides.

   The symbol match is exact on purpose. Ethereum is full of impersonation
   tokens: one address in that test carried 'UЅDТ' with a Cyrillic Ѕ, '𝐔𝐒𝐃𝐓'
   in mathematical bold, and a token named 'Visit https://usd-coin.net to claim
   rewards'. A fuzzy or case-folded match would feed the model fake USDT. */
const ETHERSCAN = "https://api.etherscan.io/v2/api";
const ETHERSCAN_KEY = "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ";
// chainid -> the chains this key reaches on the free tier.
export const EVM_CHAINS = {Ethereum: 1, Arbitrum: 42161, Polygon: 137};
const PAGE_SIZE = 200;

function normaliseEvm(rows) {
  const out = [];
  for (const r of rows || []) {
    const sym = (r.tokenSymbol || "").toUpperCase();
    if (!STABLES.has(sym)) continue;            // exact match, see above
    const dec = Number(r.tokenDecimal ?? 18);
    const raw = Number(r.value);
    if (!isFinite(raw) || !isFinite(dec)) continue;
    const usd = raw / Math.pow(10, dec);
    if (!(usd > 0)) continue;                   // mirrors WHERE amount_usd > 0
    const t = Number(r.timeStamp) * 1000;
    if (!r.from || !r.to || !isFinite(t)) continue;
    out.push({from: r.from.toLowerCase(), to: r.to.toLowerCase(), usd, t});
  }
  return out;
}

/* Two directions, for the same reason the Tron path has two.

   DESCENDING for behaviour. What an address is doing now is what the model
   reads, and a busy wallet's recent pages are where that lives.

   ASCENDING for age, separately. Sorting the whole history ascending does not
   work on its own: Binance's earliest token transfers are FTM, XDATA and COTI,
   so two ascending pages of 200 contain no stablecoin at all and the address
   reads as having no history. Age instead comes from one tiny query per major
   stablecoin, which returns that token's first transfer directly.

   Age is measured from the first STABLECOIN transfer, because that is what the
   feature layer measures: its legs table is built from stablecoin rows only. */
const AGE_TOKENS = [
  "0xdAC17F958D2ee523a2206206994597C13D831ec7",   // USDT
  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",   // USDC
];

function esUrl(chainId, params) {
  const q = Object.entries(params).map(([k, v]) =>
    `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join("&");
  return `${ETHERSCAN}?chainid=${chainId}&apikey=${ETHERSCAN_KEY}&${q}`;
}

async function esGet(url, signal) {
  let res = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    res = await fetch(url, {signal});
    if (res.status !== 429) break;
    await sleep(900 * Math.pow(2, attempt));
  }
  if (!res || res.status === 429) throw new Error("rate_limited");
  if (!res.ok) throw new Error("fetch_failed");
  const j = await res.json();
  if (j.status !== "1" && /rate limit|max calls/i.test(String(j.result || j.message || "")))
    throw new Error("rate_limited");
  return Array.isArray(j.result) ? j.result : [];
}

/** ERC-20 stablecoin history, newest first. Mirrors fetchTransfers. */
export async function fetchTransfersEvm(address, {maxPages = 6, signal, chain = "Ethereum"} = {}) {
  const chainId = EVM_CHAINS[chain];
  if (!chainId) throw new Error("unsupported_chain");
  const out = [];
  let hitCap = true;
  for (let page = 1; page <= maxPages; page++) {
    const rows = await esGet(esUrl(chainId, {
      module: "account", action: "tokentx", address,
      page, offset: PAGE_SIZE, sort: "desc"}), signal);
    out.push(...normaliseEvm(rows));
    if (rows.length < PAGE_SIZE) { hitCap = false; break; }
  }
  Object.defineProperty(out, "truncated", {value: hitCap, enumerable: false});
  return out;
}

/** Earliest stablecoin transfer, in one small query per token. */
export async function fetchFirstSeenEvm(address, {signal, chain = "Ethereum"} = {}) {
  const chainId = EVM_CHAINS[chain];
  if (!chainId) return null;
  let first = null;
  for (const token of AGE_TOKENS) {
    try {
      const rows = await esGet(esUrl(chainId, {
        module: "account", action: "tokentx", address,
        contractaddress: token, page: 1, offset: 1, sort: "asc"}), signal);
      const r = rows[0];
      if (!r) continue;
      const t = Number(r.timeStamp) * 1000;
      if (isFinite(t) && (first === null || t < first)) first = t;
    } catch (e) {
      if (e && e.message === "rate_limited") throw e;
      // A token this address never touched is not an error.
    }
  }
  return first;
}

/** True only for a checksum-shaped EVM address. */
export function isValidEvmAddress(a) {
  return typeof a === "string" && /^0x[0-9a-fA-F]{40}$/.test(a);
}
