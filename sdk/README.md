# @veridis/sdk

```ts
import { VeridisClient } from "@veridis/sdk";

const veridis = new VeridisClient({ baseUrl: "https://api.veridis.dev", timeoutMs: 400 });

const risk = await veridis.scoreOrFailOpen({
  chain: "tron",
  sender: senderAddress,
  destination: destinationAddress,
  amountUsd: 3000,
});

if (risk.band === "high_risk") {
  showInterstitial({ title: "This looks like a scam", reasons: risk.reasons });
}
```

`scoreOrFailOpen` never throws: if the API is slow or down it resolves to the
`safe` band. A risk check must not be able to block a legitimate payment.
