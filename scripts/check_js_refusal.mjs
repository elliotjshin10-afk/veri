/* The browser half of the Etherscan refusal guard.
 *
 * Etherscan answers an exceeded rate limit with HTTP 200 and a body of
 * {status: "0", result: "Max calls per sec rate limit reached (3/sec)"}. The
 * shipped fetcher must retry that and must never pass it off as an empty
 * history - a refusal read as "nothing received yet" is how a frozen collector
 * came back "ordinary", and how a busy exchange wallet came back with 0 payers.
 *
 * Driven against site/scorer.js itself, not a copy, so the file that ships is
 * the file under test. Run: node scripts/check_js_refusal.mjs
 */
import {fetchTransfersEvm, fetchFirstSeenEvm} from "../site/scorer.js";

const REFUSAL = {status: "0", message: "NOTOK",
                 result: "Max calls per sec rate limit reached (3/sec)"};
const EMPTY = {status: "0", message: "No transactions found",
               result: "No transactions found"};

const USDT = "0xdAC17F958D2ee523a2206206994597C13D831ec7";

function rows(n) {
  return Array.from({length: n}, (_, i) => ({
    contractAddress: USDT,
    tokenSymbol: "USDT", tokenDecimal: "6", value: "1000000",
    timeStamp: String(1600000000 + i), hash: "0x" + String(i).padStart(64, "0"),
    from: "0x" + String(i).padStart(40, "0"), to: "0x" + "9".repeat(40),
  }));
}

let calls = [];
function serve(queue) {
  calls = [];
  globalThis.fetch = async (url) => {
    calls.push(String(url));
    const body = queue.length ? queue.shift() : {status: "1", result: []};
    return {ok: true, status: 200, json: async () => body};
  };
}

const fails = [];
function check(name, ok, detail) {
  if (ok) console.log("  ok   " + name);
  else { console.log("  FAIL " + name + (detail ? " - " + detail : "")); fails.push(name); }
}

const PAGE = 1000;
const A = "0x" + "a".repeat(40);

/* 1. A refusal is retried, and the retry's data is returned. */
serve([REFUSAL, {status: "1", result: rows(3)}]);
let tx = await fetchTransfersEvm(A, {maxPages: 1});
check("a refusal is retried rather than returned", tx.length === 3 && calls.length === 2,
      `rows=${tx.length} calls=${calls.length}`);

/* 2. A refusal that never clears throws, and throws rate_limited - not an
 *    empty history, and not a generic failure the page would word wrongly. */
serve([REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL]);
let err = null;
try { await fetchTransfersEvm(A, {maxPages: 1}); } catch (e) { err = e; }
check("a persistent refusal throws rate_limited", err && err.message === "rate_limited",
      err ? err.message : "did not throw");

/* 3. A documented empty history is empty, not an error. */
serve([EMPTY]);
tx = await fetchTransfersEvm(A, {maxPages: 2});
check("an empty history returns empty", tx.length === 0 && tx.truncated === false,
      `rows=${tx.length} truncated=${tx.truncated}`);

/* 4. A refusal on page two does not quietly shorten the history. */
serve([{status: "1", result: rows(PAGE)}, REFUSAL, REFUSAL, REFUSAL, REFUSAL]);
err = null;
try { await fetchTransfersEvm(A, {maxPages: 3}); } catch (e) { err = e; }
check("a refusal mid-history throws instead of truncating",
      err && err.message === "rate_limited", err ? err.message : "returned page 1 only");

/* 5. A full page means the budget may be exhausted; a short page means done. */
serve([{status: "1", result: rows(PAGE)}, {status: "1", result: rows(PAGE)}]);
tx = await fetchTransfersEvm(A, {maxPages: 2});
check("exhausting the page budget is reported as truncated", tx.truncated === true,
      `truncated=${tx.truncated}`);
serve([{status: "1", result: rows(PAGE - 1)}]);
tx = await fetchTransfersEvm(A, {maxPages: 3});
check("a short page completes the history", tx.truncated === false,
      `truncated=${tx.truncated}`);

/* 6. The age probe must propagate a refusal. Returning null would hand the
 *    feature layer an invented age for a truncated history. */
serve([REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL, REFUSAL]);
err = null;
try { await fetchFirstSeenEvm(A); } catch (e) { err = e; }
check("the age probe propagates a refusal", err && err.message === "rate_limited",
      err ? err.message : "swallowed the refusal");

/* 7. Requests are paced far enough apart for a 3/sec cap. */
serve([{status: "1", result: rows(PAGE)}, {status: "1", result: rows(PAGE)},
       {status: "1", result: rows(PAGE)}]);
const t0 = Date.now();
await fetchTransfersEvm(A, {maxPages: 3});
const per = (Date.now() - t0) / 2;          // gaps between 3 requests
check("requests are paced under 3/sec", per >= 333, `${per.toFixed(0)}ms between requests`);

/* 8. Two histories fetched AT ONCE stay paced. A two-sided check reads the
 *    sender's and the destination's together, and per-caller pacing lets both
 *    fire in the same millisecond - the one case where the gap matters most. */
let stamps = [];
globalThis.fetch = async (url) => {
  stamps.push(Date.now());
  return {ok: true, status: 200,
          json: async () => ({status: "1", result: rows(3)})};
};
stamps = [];
await Promise.all([
  fetchTransfersEvm("0x" + "b".repeat(40), {maxPages: 2}),
  fetchTransfersEvm("0x" + "c".repeat(40), {maxPages: 2}),
]);
stamps.sort((a, b) => a - b);
let tightest = Infinity;
for (let i = 1; i < stamps.length; i++) tightest = Math.min(tightest, stamps[i] - stamps[i - 1]);
check("concurrent fetches share one pacing queue",
      stamps.length >= 2 && tightest >= 333,
      `${stamps.length} requests, tightest gap ${tightest}ms`);

/* 9. A counterfeit token is not a stablecoin, whatever it calls itself.
 *    Anyone can deploy a contract whose symbol is the real ASCII "USDT". One
 *    such transfer claimed $9e39, and every address that received one cleared
 *    the institutional guard on that fake inflow, so the page called a
 *    collection point an exchange. The contract decides, not the label. */
const fakeRow = {
  contractAddress: "0x" + "f".repeat(40), tokenSymbol: "USDT",
  tokenDecimal: "0", value: "9" + "0".repeat(39), timeStamp: "1700000000",
  hash: "0xdead", from: "0x" + "c".repeat(40), to: "0x" + "9".repeat(40),
};
serve([{status: "1", result: [...rows(2), fakeRow]}]);
tx = await fetchTransfersEvm(A, {maxPages: 1});
check("a counterfeit USDT is rejected on its contract",
      tx.length === 2 && tx.every(r => r.usd < 1e6),
      `kept ${tx.length} rows, max usd ${Math.max(0, ...tx.map(r => r.usd))}`);

console.log(fails.length
  ? `\n${fails.length} refusal check(s) FAILED`
  : "\nall refusal checks passed");
process.exit(fails.length ? 1 : 0);
