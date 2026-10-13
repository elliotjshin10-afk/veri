/* The usage counter, as a Cloudflare Worker.
 *
 * Same contract as scripts/tally_server.py, so the page does not care which one
 * is behind the endpoint: POST /t to record, GET /stats.json to read. The
 * Python one is for a laptop or a VPS; this one is for a free URL with no
 * server to run, which is what a static site on GitHub Pages needs.
 *
 * Signups live here too, at POST /email. They are never readable without a
 * READ_KEY variable, because an aggregate count is fine to publish and a list
 * of people's addresses is not.
 *
 * Setup, once, no CLI required:
 *   1. dash.cloudflare.com -> Workers & Pages -> Create -> Worker -> paste this
 *   2. Settings -> Variables -> KV Namespace Bindings -> add one named TALLY
 *      (create the namespace on that screen if you have none)
 *   3. Deploy, copy the workers.dev URL
 *   4. Put that URL in site/index.html's veridis-tally meta tag, then `make pages`
 *
 * It stores the same four fields and refuses the rest, for the same reason: the
 * record is built key by key rather than filtered, so a client that starts
 * sending an address cannot get one persisted. Cloudflare's free KV allows
 * 1,000 writes a day, so counts are kept as daily aggregates rather than one
 * row per event. That is enough to say how many people checked how many
 * addresses and what came back, which is the whole question.
 */

const BANDS = new Set(["ordinary", "elevated", "high"]);
const CHAINS = new Set(["tron", "ethereum"]);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "content-type",
  "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
};

function clean(raw) {
  if (!raw || (raw.e !== "check" && raw.e !== "view")) return null;
  const out = { e: raw.e };
  // A browser id only counts if it looks like the one we mint. Anything else is
  // noise or probing, and neither should widen what gets stored.
  if (typeof raw.u === "string" && /^[a-z0-9]{8,40}$/i.test(raw.u)) out.u = raw.u;
  if (raw.e === "check") {
    if (!BANDS.has(raw.b) || !CHAINS.has(raw.c)) return null;
    out.b = raw.b;
    out.c = raw.c;
    out.w = raw.w ? 1 : 0;
  }
  return out;
}

const today = () => new Date().toISOString().slice(0, 10);

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/$/, "");

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }

    if (request.method === "GET" && (path === "/stats.json" || path === "/stats")) {
      const list = await env.TALLY.list({ prefix: "d:" });
      const days = {};
      let checks = 0, views = 0, withWallet = 0;
      const bands = {}, chains = {};
      for (const k of list.keys) {
        const v = JSON.parse((await env.TALLY.get(k.name)) || "{}");
        const day = k.name.slice(2);
        days[day] = v.checks || 0;
        checks += v.checks || 0;
        views += v.views || 0;
        withWallet += v.with_wallet || 0;
        for (const [b, n] of Object.entries(v.bands || {})) bands[b] = (bands[b] || 0) + n;
        for (const [c, n] of Object.entries(v.chains || {})) chains[c] = (chains[c] || 0) + n;
      }
      const browsers = JSON.parse((await env.TALLY.get("browsers")) || "[]");
      const body = JSON.stringify({
        checks, views, browsers: browsers.length, bands, chains,
        with_wallet: withWallet, by_day: days,
      }, null, 2);
      return new Response(body, {
        headers: { ...CORS, "content-type": "application/json" },
      });
    }

    /* Launch-list signups. Kept in the same Worker because standing up a second
     * service to hold a few hundred email addresses is not worth the surface.
     *
     * There is deliberately no public read. Counts can be public, a list of
     * people's email addresses cannot, so reading requires a secret set as a
     * Worker variable and the endpoint simply does not exist until one is. */
    if (request.method === "POST" && path === "/email") {
      let addr = null, who = null;
      try {
        const b = await request.json();
        if (typeof b.email === "string" && b.email.length <= 200
            && /^[^@\s]+@[^@\s.]+\.[^@\s]+$/.test(b.email)) addr = b.email.toLowerCase();
        if (typeof b.who === "string" && b.who.length <= 60) who = b.who;
      } catch (e) { /* not an email then */ }
      if (addr) {
        // Keyed by address, so a double submit is one row rather than two.
        await env.TALLY.put("e:" + addr, JSON.stringify(
          { t: new Date().toISOString(), who: who || "" }));
      }
      return new Response(null, { status: 204, headers: CORS });
    }

    if (request.method === "GET" && path === "/emails.json") {
      const secret = env.READ_KEY;
      if (!secret || url.searchParams.get("key") !== secret) {
        return new Response(null, { status: 404, headers: CORS });
      }
      const list = await env.TALLY.list({ prefix: "e:" });
      const out = [];
      for (const k of list.keys) {
        const v = JSON.parse((await env.TALLY.get(k.name)) || "{}");
        out.push({ email: k.name.slice(2), when: v.t, who: v.who });
      }
      return new Response(JSON.stringify(out, null, 2), {
        headers: { ...CORS, "content-type": "application/json" },
      });
    }

    if (request.method === "POST" && path === "/t") {
      let ev = null;
      try {
        ev = clean(await request.json());
      } catch (e) { /* malformed body is simply not an event */ }
      if (ev) {
        const key = "d:" + today();
        const cur = JSON.parse((await env.TALLY.get(key)) || "{}");
        cur.bands = cur.bands || {};
        cur.chains = cur.chains || {};
        if (ev.e === "view") {
          cur.views = (cur.views || 0) + 1;
        } else {
          cur.checks = (cur.checks || 0) + 1;
          cur.with_wallet = (cur.with_wallet || 0) + ev.w;
          cur.bands[ev.b] = (cur.bands[ev.b] || 0) + 1;
          cur.chains[ev.c] = (cur.chains[ev.c] || 0) + 1;
        }
        await env.TALLY.put(key, JSON.stringify(cur));
        if (ev.u) {
          // Capped deliberately. The question is "roughly how many people",
          // not "exactly which ones", and an unbounded list is both a cost and
          // a thing that starts to resemble a record of individuals.
          const seen = JSON.parse((await env.TALLY.get("browsers")) || "[]");
          if (!seen.includes(ev.u) && seen.length < 5000) {
            seen.push(ev.u);
            await env.TALLY.put("browsers", JSON.stringify(seen));
          }
        }
      }
      // 204 whatever happened: a counter must not report what it rejected, and
      // must never be a reason the page behaves differently.
      return new Response(null, { status: 204, headers: CORS });
    }

    return new Response(null, { status: 404, headers: CORS });
  },
};
