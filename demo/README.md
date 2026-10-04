# Replay demo

`make demo` generates `replay.json` and `index.html` here from **real held-out
chain data only** — one real victim's real transfer sequence, replayed step by
step against a real public-blocklist timeline.

Nothing is committed to this directory on purpose. A replay page built from
anything other than real, verifiable on-chain data would be worse than useless
in front of a partner: every address and transaction hash on the page links to
Tronscan precisely so it can be checked in the room.

Generating it requires a trained model, because the demo replays only test-set
victims — showing the model recognising a case it was trained on would prove
nothing:

```bash
make features && make train && make demo
```
