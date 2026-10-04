/**
 * Veridis pre-send risk SDK.
 *
 * Call this before an irreversible transfer executes. The consuming app
 * decides what to show; this returns a number and human-readable reasons.
 */

export type Band = "safe" | "caution" | "high_risk";

export interface ScoreRequest {
  chain: string;
  sender: string;
  destination: string;
  amountUsd: number;
  asset?: string;
}

export interface ScoreResponse {
  score: number;
  band: Band;
  reasons: string[];
  modelVersion: string;
  computedAt: string;
  latencyMs: number;
  featuresAvailable: boolean;
}

export interface ClientOptions {
  baseUrl?: string;
  apiKey?: string;
  /** Hard ceiling on the call. A send must never block on a slow risk check. */
  timeoutMs?: number;
  /** Returned when the API is unreachable or times out. Default "safe". */
  failOpenBand?: Band;
}

export class VeridisError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "VeridisError";
  }
}

export class VeridisClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly failOpenBand: Band;

  constructor(opts: ClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? 400;
    this.failOpenBand = opts.failOpenBand ?? "safe";
  }

  async score(req: ScoreRequest): Promise<ScoreResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(`${this.baseUrl}/score`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "content-type": "application/json",
          ...(this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {}),
        },
        body: JSON.stringify({
          chain: req.chain,
          sender: req.sender,
          destination: req.destination,
          amount_usd: req.amountUsd,
          asset: req.asset ?? "USDT",
        }),
      });
      if (!res.ok) {
        throw new VeridisError(`score failed: ${res.status}`, res.status);
      }
      const j = await res.json();
      return {
        score: j.score,
        band: j.band as Band,
        reasons: j.reasons ?? [],
        modelVersion: j.model_version,
        computedAt: j.computed_at,
        latencyMs: j.latency_ms,
        featuresAvailable: j.features_available,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Never let a risk check break a send. On timeout or outage this resolves to
   * `failOpenBand` rather than throwing - failing closed would mean a wallet
   * outage blocks every legitimate payment.
   */
  async scoreOrFailOpen(req: ScoreRequest): Promise<ScoreResponse> {
    try {
      return await this.score(req);
    } catch {
      return {
        score: 0,
        band: this.failOpenBand,
        reasons: [],
        modelVersion: "unavailable",
        computedAt: new Date().toISOString(),
        latencyMs: this.timeoutMs,
        featuresAvailable: false,
      };
    }
  }
}
