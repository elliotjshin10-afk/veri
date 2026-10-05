"""M5 - SHAP-driven reason codes.

The warning text is the product. A generic warning gets clicked through; a
specific, checkable statement does not. Every template below states something
the user can verify on a block explorer in under a minute - that constraint is
what keeps the warnings honest.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# feature -> (template, direction, guard)
# direction: +1 means the feature pushes risk up when large; -1 when small.
# guard decides whether the statement is actually true and worth showing.
TEMPLATES: dict[str, dict] = {
    "dest_age_days": {
        "text": "Destination address first seen {value:.0f} days ago",
        "guard": lambda v, r: v is not None and v <= 90,
    },
    "dest_senders_7d": {
        "text": "{value:.0f} unrelated wallets have sent to it in the past 7 days",
        "guard": lambda v, r: v is not None and v >= 3,
    },
    "dest_senders_30d": {
        "text": "{value:.0f} unrelated wallets have sent to it in the past 30 days",
        "guard": lambda v, r: v is not None and v >= 5,
    },
    "dest_median_hold_secs": {
        "text": "Funds sent here are forwarded onward within {mins} of arriving",
        "guard": lambda v, r: v is not None and 0 <= v <= 3600,
    },
    "dest_consolidation_ratio": {
        "text": "{pct:.0f}% of everything this address forwards goes to a single address",
        "guard": lambda v, r: v is not None and v >= 0.6,
    },
    "dest_forward_ratio": {
        "text": "This address keeps almost nothing it receives - {pct:.0f}% is forwarded out",
        # Capped at 1.0: a ratio above 1 just means it also forwards funds that
        # arrived before our window, and ">120% forwarded" would read as nonsense.
        "guard": lambda v, r: v is not None and 0.85 <= v <= 1.5,
    },
    "dest_distinct_out": {
        "text": "Everything it receives is forwarded to only {value:.0f} address(es)",
        "guard": lambda v, r: v is not None and 0 < v <= 3,
    },
    "is_first_send_to_dest": {
        "text": "You have never sent to this address before",
        "guard": lambda v, r: v == 1,
    },
    "escalation_ratio": {
        "text": "This transfer is {value:.0f}x larger than your previous one to it",
        "guard": lambda v, r: v is not None and v >= 2,
    },
    "amount_vs_sender_median": {
        "text": "This transfer is {value:.0f}x your typical transfer size",
        "guard": lambda v, r: v is not None and v >= 3,
    },
    "amount_vs_balance": {
        "text": "This transfer moves {pct:.0f}% of your remaining balance",
        "guard": lambda v, r: v is not None and 0.5 <= v <= 1.5,
    },
    "sender_burst_24h": {
        "text": "You have made {value:.0f} transfers in the past 24 hours",
        "guard": lambda v, r: v is not None and v >= 3,
    },
    "first_send_to_young_dest": {
        "text": "First transfer to an address that is less than 30 days old",
        "guard": lambda v, r: v == 1,
    },
    "no_shared_counterparties": {
        "text": "This address has no connection to anyone you have dealt with before",
        "guard": lambda v, r: v == 1,
    },
    "dest_usd_per_sender": {
        "text": "Wallets that send here average ${value:,.0f} each before stopping",
        "guard": lambda v, r: v is not None and v >= 500,
    },
    "dest_funder_fanout": {
        "text": "The wallet that funded this address has also funded {value:.0f} others",
        "guard": lambda v, r: v is not None and v >= 5,
    },
    "pair_secs_since_last": {
        "text": "Your last transfer to this address was {mins} ago",
        "guard": lambda v, r: v is not None and 0 <= v <= 86400,
    },
}


def _humanise_secs(secs: float) -> str:
    secs = max(secs, 0)
    if secs < 90:
        return f"{secs:.0f} seconds"
    if secs < 5400:
        return f"{secs/60:.0f} minutes"
    if secs < 172800:
        return f"{secs/3600:.0f} hours"
    return f"{secs/86400:.0f} days"


def render(feature: str, value: float, row: dict) -> str | None:
    spec = TEMPLATES.get(feature)
    if spec is None or value is None:
        return None
    try:
        if not spec["guard"](value, row):
            return None
    except Exception:
        return None
    return spec["text"].format(
        value=value,
        pct=min(value, 1.0) * 100 if value <= 1.5 else value,
        mins=_humanise_secs(value),
    )


class ReasonEngine:
    """Maps the top SHAP contributors for one scored event to warning text."""

    def __init__(self, booster, feature_names: list[str]) -> None:
        self.booster = booster
        self.feature_names = feature_names
        self._explainer = None

    def _shap(self, X: np.ndarray) -> np.ndarray:
        # LightGBM computes exact tree SHAP natively; avoids a shap dependency
        # on the serving path while giving identical values.
        contrib = self.booster.predict(X, pred_contrib=True)
        return np.asarray(contrib)[:, :-1]  # drop the bias column

    def explain(self, X: np.ndarray, rows: list[dict], top_n: int = 5) -> list[list[str]]:
        contrib = self._shap(X)
        out: list[list[str]] = []
        for i, row in enumerate(rows):
            order = np.argsort(-contrib[i])  # most risk-increasing first
            reasons: list[str] = []
            for j in order:
                if contrib[i][j] <= 0:
                    break
                name = self.feature_names[j]
                text = render(name, row.get(name), row)
                if text and text not in reasons:
                    reasons.append(text)
                if len(reasons) >= top_n:
                    break
            out.append(reasons)
        return out

    def reassure(self, X: np.ndarray, rows: list[dict], top_n: int = 3) -> list[list[str]]:
        """The mirror of explain(): what held the score DOWN on this event.

        Same contributions, read from the other end. A log that says "stopped"
        and "went through" has to account for both, and the honest account of
        "went through" is the feature that actually argued for it.
        """
        contrib = self._shap(X)
        out: list[list[str]] = []
        for i, row in enumerate(rows):
            order = np.argsort(contrib[i])  # most risk-DECREASING first
            texts: list[str] = []
            for j in order:
                if contrib[i][j] >= 0:
                    break
                name = self.feature_names[j]
                text = render_clear(name, row.get(name), row)
                if text and text not in texts:
                    texts.append(text)
                if len(texts) >= top_n:
                    break
            out.append(texts)
        return out


# The templates above only ever explain a WARNING. A log of past decisions has
# to explain the other verdict too - why a transfer went through - and "no
# reasons fired" is not an explanation. These are the mirror image: statements
# that are true when the feature pushed the score DOWN, rendered from the same
# SHAP contributions so the sentence names the thing that actually decided it.
CLEAR_TEMPLATES: dict[str, dict] = {
    "sender_prior_sends_to_dest": {
        "text": "This wallet had already paid the address {value:.0f} times",
        "guard": lambda v, r: v is not None and v >= 1,
    },
    "dest_forward_ratio": {
        "text": "The address keeps most of what it receives instead of forwarding it on",
        "guard": lambda v, r: v is not None and 0 <= v < 0.7,
    },
    "dest_senders_30d": {
        "text": "Only {value:.0f} wallets paid it in the previous 30 days",
        "guard": lambda v, r: v is not None and v <= 2,
    },
    "dest_senders_all": {
        "text": "Only {value:.0f} wallets had ever paid it",
        "guard": lambda v, r: v is not None and 0 < v <= 25,
    },
    "dest_median_hold_secs": {
        "text": "Money that arrives here sits for {mins} rather than moving straight on",
        "guard": lambda v, r: v is not None and v >= 86_400,
    },
    "dest_age_days": {
        "text": "The address had been active for {value:,.0f} days",
        "guard": lambda v, r: v is not None and v >= 180,
    },
    "escalation_ratio": {
        "text": "No escalation - about the size of the last transfer to it",
        "guard": lambda v, r: v is not None and 0 < v <= 1.5,
    },
    "amount_vs_sender_median": {
        "text": "The amount was ordinary for this wallet",
        "guard": lambda v, r: v is not None and 0.2 <= v <= 2.0,
    },
    "shared_counterparties": {
        "text": "{value:.0f} counterparties in common with wallets it already dealt with",
        "guard": lambda v, r: v is not None and v >= 1,
    },
    "pair_span_days": {
        "text": "The two addresses had been transacting for {value:,.0f} days",
        "guard": lambda v, r: v is not None and v >= 30,
    },
}


def render_clear(feature: str, value: float, row: dict) -> str | None:
    spec = CLEAR_TEMPLATES.get(feature)
    if spec is None or value is None:
        return None
    try:
        if not spec["guard"](value, row):
            return None
    except Exception:
        return None
    return spec["text"].format(
        value=value,
        pct=min(value, 1.0) * 100 if value <= 1.5 else value,
        mins=_humanise_secs(value),
    )
