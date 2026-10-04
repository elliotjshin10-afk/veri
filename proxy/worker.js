/**
 * TronGrid proxy — Cloudflare Worker.
 *
 * The site is a static page, so a key placed in its JavaScript is readable by
 * anyone who opens devtools. It would be scraped, used elsewhere, and either
 * rate-limited into uselessness or revoked. This keeps the key server-side and
 * gives the page an endpoint that looks the same.
 *
 * Deploy:
 *   npm i -g wrangler
 *   wrangler deploy
 *   wrangler secret put TRONGRID_API_KEY      # paste the key when prompted
 *
 * Then set PROXY_BASE in site/index.html to the Worker URL.
 *
 * Free tier is 100,000 requests/day, which is far more than a launch needs.
 */

// Only these TronGrid paths are reachable. An open proxy would let anyone use
// the key for anything, which is the problem we are trying to avoid.
const ALLOWED = [
  /^\/v1\/accounts\/T[1-9A-HJ-NP-Za-km-z]{33}\/transactions\/trc20$/,
  /^\/v1\/accounts\/T[1-9A-HJ-NP-Za-km-z]{33}$/,
];

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, OPTIONS",
  "access-control-allow-headers": "content-type",
  "access-control-max-age": "86400",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, {headers: CORS});
    if (request.method !== "GET") {
      return json({error: "method_not_allowed"}, 405);
    }

    const url = new URL(request.url);
    if (!ALLOWED.some((re) => re.test(url.pathname))) {
      return json({error: "path_not_allowed"}, 403);
    }

    const target = new URL("https://api.trongrid.io" + url.pathname + url.search);
    let upstream;
    try {
      upstream = await fetch(target, {
        headers: {
          "TRON-PRO-API-KEY": env.TRONGRID_API_KEY,
          accept: "application/json",
        },
        // Identical requests are common (shared links, retries); a short cache
        // keeps the key's quota for requests that actually need it.
        cf: {cacheTtl: 60, cacheEverything: true},
      });
    } catch {
      return json({error: "upstream_unreachable"}, 502);
    }

    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: {
        ...CORS,
        "content-type": "application/json;charset=utf-8",
        "cache-control": "public, max-age=60",
      },
    });
  },
};

function json(obj, status) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: {...CORS, "content-type": "application/json"},
  });
}
