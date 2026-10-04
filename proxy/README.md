# TronGrid proxy

The site is static, so a key in its JavaScript is public. This holds the key
server-side and exposes the two TronGrid paths the page needs.

```bash
npm i -g wrangler
cd proxy
wrangler deploy
wrangler secret put TRONGRID_API_KEY     # paste the key
```

Then in `site/index.html`:

```js
const PROXY_BASE = "https://veridis-trongrid-proxy.<you>.workers.dev";
```

Left empty, the site calls TronGrid directly with no key — which still works,
just on the keyless rate limit.

**It is not an open proxy.** Only `/v1/accounts/{tron address}/transactions/trc20`
and `/v1/accounts/{tron address}` pass; anything else gets a 403. Responses are
cached 60s, so repeat lookups of the same address cost nothing.

Free tier: 100,000 requests/day.
