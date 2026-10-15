# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

- **Subtle feature skew / distribution shift.** `ctx.baseline` bounds sit at
  clean mean + 3σ, so a "subtle" fault deliberately lands *under* them. A
  static check can't tell it apart from noise. I added a second layer: a robust
  running estimate (median/MAD) of each metric, built only from events I judged
  clean, and an alert when the upper-tail robust z goes above 3. The baseline
  bound sets a minimum on the spread, so a tight early history can't turn
  noise into alerts.
- **Volume / mean-amount drift.** These baselines are two-sided, so μ and σ
  come straight from them (μ = midpoint, σ = width/6). I alert at |z| > 2.6
  rather than 3, because one missed fault costs about 3–5× one false alarm
  under the scoring formula.
- **Missing upstream lineage edges.** The tool returns only the *actual*
  upstream set. So the check compares against a declared set when the payload
  carries one. Otherwise it uses the dominant clean upstream set learned for
  that job, and fires only once that set is clearly dominant (≥3 sightings and
  ≥60% of history).
- **Faults that move several fields a little** (for example, a batch that is
  slightly stale *and* slightly short). These are combined as a χ²-style sum of
  squared z-scores, which needs at least two fields to be moving.
- Contract schema/type violations and orphan outputs (0 downstream) are
  deterministic flags, so they're easy.

Practice result: score 48.97 / 50 possible, TPR 1.00 (33/33, including both
subtle feature-skew faults), FPR 0.034 (3 false alarms out of 87 clean
events), 180 of 220 credits spent. I kept the cut-offs where they are rather
than trimming those 3 false alarms, because private leans towards subtle
faults and a missed fault costs more than a false alarm.

Private result: score 31.79, TPR 0.65, FPR 0.02, no cost overage. The drop in
TPR versus practice confirms the point above: about a third of private's
faults sit inside what my detector treats as normal variance. The low FPR
suggests I could have afforded more aggressive cut-offs.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

Each event gets exactly one metered call, the only way to see any signal, and
no redundant calls. On private that's about 300 credits against a 320 budget,
so I always pay for coverage. Skipping an event saves ~1.5 credits
(≈0.1 score points of overage), but it risks a missed fault worth ~1 point.
With another pass I would:

1. Calibrate `Z_KNOWN`, `Z_ONLINE` and the χ² cut-off against the practice
   answer key, using per-fault-type ROC curves instead of one global
   cut-off.
2. Keep history per entity (per feature view, per corpus, per contract), not
   pooled, once there are enough samples. Different views have different noise
   floors.
3. Warm up the online estimators faster: for the first ~8 events of each type
   only the static bounds apply, so early subtle faults are missed.
