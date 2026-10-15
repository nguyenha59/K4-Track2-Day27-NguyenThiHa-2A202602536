"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.

Strategy
--------
* One metered call per event (the only source of signal) — no redundant calls.
* Two layers per metric:
  1. static: ctx.baseline bounds (clean mean ± 3σ). Two-sided bounds give us
     μ and σ directly, so we test those with a slightly tighter z cut-off.
  2. adaptive: a robust running estimate (median / MAD) of each metric over
     events we judged clean so far, so faults sitting just under the static
     bound but far outside the stream's actual spread still get flagged.
* Several mildly-elevated metrics on one event are combined (sum of squared
  z-scores) to catch faults that move more than one field a little.
* Scoring asymmetry: one missed fault costs ~3–5x what one false alarm costs,
  so cut-offs lean towards alerting.
"""
import math
from api import Verdict

# z cut-off for metrics whose μ/σ are known from a two-sided baseline
Z_KNOWN = 2.6
# robust-z cut-off for metrics tracked online (one-sided, upper tail)
Z_ONLINE = 3.0
# minimum clean samples before an online estimate is trusted
MIN_SAMPLES = 8
# combined evidence: sum of squared positive z-scores across fields
COMBINED_CHI2 = 9.0


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# ---------------------------------------------------------------- helpers

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _bounds(ctx, lo_key, hi_key):
    """μ, σ implied by a mean ± 3σ baseline pair."""
    lo, hi = ctx.baseline.get(lo_key), ctx.baseline.get(hi_key)
    if not (_num(lo) and _num(hi)) or hi <= lo:
        return None
    return (lo + hi) / 2.0, (hi - lo) / 6.0


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _robust_z(ctx, name, value, upper_bound=None):
    """Upper-tail robust z of value vs. the clean history of `name`.
    If a one-sided baseline max (≈ μ+3σ) is known, it floors the spread so a
    very tight early history can't turn noise into alerts."""
    hist = ctx.state.get("hist", {}).get(name, [])
    if len(hist) < MIN_SAMPLES:
        return None
    med = _median(hist)
    mad = _median([abs(h - med) for h in hist]) * 1.4826
    floor = 1e-9
    if upper_bound is not None and upper_bound > med:
        # never assume a spread tighter than 1/6 of the gap to the 3σ bound
        floor = max(floor, (upper_bound - med) / 6.0)
    sigma = max(mad, floor, abs(med) * 0.01)
    return (value - med) / sigma


def _remember(ctx, name, value, cap=400):
    hist = ctx.state.setdefault("hist", {}).setdefault(name, [])
    hist.append(value)
    if len(hist) > cap:
        del hist[0]


def _upper_check(ctx, name, value, bound_key, reasons, zs):
    """One-sided (larger = worse) metric: static bound + online robust z."""
    if not _num(value):
        return
    bound = ctx.baseline.get(bound_key)
    if _num(bound) and value > bound:
        reasons.append(f"{name}={value:.4g}>{bound_key}")
        zs.append(9.0)
        return
    z = _robust_z(ctx, name, value, bound if _num(bound) else None)
    if z is not None:
        if z > Z_ONLINE:
            reasons.append(f"{name} robust_z={z:.2f}")
        if z > 0:
            zs.append(z * z)


def _two_sided_check(ctx, name, value, lo_key, hi_key, reasons, zs):
    if not _num(value):
        return
    mb = _bounds(ctx, lo_key, hi_key)
    if mb is None:
        return
    mu, sd = mb
    z = abs(value - mu) / sd if sd > 0 else 0.0
    if z > Z_KNOWN:
        reasons.append(f"{name}={value:.4g} z={z:.2f}")
    zs.append(z * z)


def _verdict(ctx, pillar, reasons, zs, observed):
    """Final decision; only feeds the online history with events judged clean
    so faults don't inflate the reference distribution."""
    # chi2-style combination of moderate deviations (needs ≥2 fields moving)
    combined = sum(z for z in zs if z > 1.0)
    moving = sum(1 for z in zs if z > 2.25)  # |z| > 1.5
    if not reasons and moving >= 2 and combined > COMBINED_CHI2:
        reasons.append(f"combined chi2={combined:.1f}")
    alert = bool(reasons)
    if not alert:
        for name, value in observed.items():
            if _num(value):
                _remember(ctx, name, value)
    return Verdict(alert=alert, pillar=pillar,
                   confidence=0.9 if alert else 0.7,
                   reason="; ".join(reasons))


def _bad(result):
    return not isinstance(result, dict) or "error" in result


# ---------------------------------------------------------------- handlers

def check_data_batch(payload, ctx):
    prof = ctx.tools.batch_profile(payload.get("batch_id"))
    if _bad(prof):
        return Verdict(alert=False, pillar="checks", reason="no profile")

    reasons, zs = [], []
    row_count = prof.get("row_count")
    nulls = prof.get("null_rate") or {}
    null_rate = nulls.get("customer_id") if isinstance(nulls, dict) else None
    mean_amount = prof.get("mean_amount")
    std_amount = prof.get("std_amount")
    staleness = prof.get("staleness_min")

    # volume (two-sided)
    _two_sided_check(ctx, "row_count", row_count, "row_count_min", "row_count_max", reasons, zs)
    # distribution: mean (two-sided, known μ/σ) + std (learned online, both tails)
    _two_sided_check(ctx, "mean_amount", mean_amount, "mean_amount_min", "mean_amount_max", reasons, zs)
    if _num(std_amount):
        z_hi = _robust_z(ctx, "std_amount", std_amount)
        z_lo = _robust_z(ctx, "neg_std_amount", -std_amount)
        for z in (z_hi, z_lo):
            if z is not None:
                if z > Z_ONLINE + 0.5:
                    reasons.append(f"std_amount={std_amount:.4g} robust_z={z:.2f}")
                if z > 0:
                    zs.append(z * z)
    # null-rate + freshness (one-sided)
    _upper_check(ctx, "null_rate", null_rate, "null_rate_max", reasons, zs)
    _upper_check(ctx, "staleness_min", staleness, "staleness_min_max", reasons, zs)
    if _num(row_count) and row_count <= 0:
        reasons.append("empty batch")

    observed = {"row_count": row_count, "null_rate": null_rate, "mean_amount": mean_amount,
                "std_amount": std_amount, "neg_std_amount": -std_amount if _num(std_amount) else None,
                "staleness_min": staleness}
    return _verdict(ctx, "checks", reasons, zs, observed)


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload.get("contract_id"), payload.get("checkpoint_batch_id"))
    if _bad(diff):
        return Verdict(alert=False, pillar="contracts", reason="no diff")

    reasons, zs = [], []
    violations = diff.get("violations") or []
    if violations:
        reasons.append("violations: " + ",".join(str(v) for v in violations))
    # SLA: freshness delay vs. declared/derived max
    delay = diff.get("freshness_delay_min")
    sla = payload.get("sla_freshness_max_min") or payload.get("freshness_sla_min")
    if _num(delay) and _num(sla) and delay > sla:
        reasons.append(f"freshness_delay={delay:.4g}>sla {sla}")
    _upper_check(ctx, "freshness_delay_min", delay, "freshness_delay_max_min", reasons, zs)
    return _verdict(ctx, "contracts", reasons, zs, {"freshness_delay_min": delay})


def _expected_upstream(payload):
    for key in ("expected_upstream", "declared_upstream", "upstream", "expected_upstreams"):
        v = payload.get(key)
        if isinstance(v, (list, tuple)):
            return v
    return None


def check_lineage_run(payload, ctx):
    sl = ctx.tools.lineage_graph_slice(payload.get("run_id"))
    if _bad(sl):
        return Verdict(alert=False, pillar="lineage", reason="no slice")

    reasons, zs = [], []
    upstream = sl.get("actual_upstream")
    upstream = list(upstream) if isinstance(upstream, (list, tuple)) else []
    down = sl.get("actual_downstream_count")
    duration = sl.get("duration_ms")

    # missing upstream edges: vs. declared upstream if the payload has one,
    # otherwise vs. the upstream set we've learned for this job.
    job = payload.get("job") or payload.get("transform") or payload.get("dataset") or "_all"
    expected = _expected_upstream(payload)
    known = ctx.state.setdefault("upstream_sets", {})
    if expected is not None:
        missing = set(map(str, expected)) - set(map(str, upstream))
        if missing:
            reasons.append("missing upstream: " + ",".join(sorted(missing)))
    else:
        typical = _typical_upstream(known.get(job))
        if typical and not typical <= set(map(str, upstream)):
            reasons.append("upstream shrank vs. history")
    if not upstream:
        reasons.append("no upstream edges")

    # orphaned output
    expected_down = payload.get("expected_downstream_count")
    if _num(down):
        if down <= 0:
            reasons.append("orphan output (0 downstream)")
        elif _num(expected_down) and down < expected_down:
            reasons.append(f"downstream {down}<{expected_down}")

    # runtime anomaly (one-sided, log scale for the online part)
    if _num(duration):
        bound = ctx.baseline.get("lineage_duration_ms_max")
        if _num(bound) and duration > bound:
            reasons.append(f"duration={duration:.0f}ms>max")
            zs.append(81.0)
        elif duration > 0:
            lb = math.log(bound) if _num(bound) and bound > 0 else None
            z = _robust_z(ctx, "log_duration", math.log(duration), lb)
            if z is not None and z > Z_ONLINE:
                reasons.append(f"duration robust_z={z:.2f}")

    v = _verdict(ctx, "lineage", reasons, zs,
                 {"log_duration": math.log(duration) if _num(duration) and duration > 0 else None})
    if not v.alert and upstream:
        sig = "|".join(sorted(map(str, upstream)))
        counts = known.setdefault(job, {})
        counts[sig] = counts.get(sig, 0) + 1
    return v


def _typical_upstream(counts):
    """The dominant clean upstream set for a job, once it's clearly dominant."""
    if not counts:
        return None
    total = sum(counts.values())
    sig, n = max(counts.items(), key=lambda kv: kv[1])
    if n < 3 or n < 0.6 * total:
        return None
    return set(sig.split("|"))


def check_feature_materialization(payload, ctx):
    fd = ctx.tools.feature_drift(payload.get("feature_view"), payload.get("batch_id"))
    if _bad(fd):
        return Verdict(alert=False, pillar="ai_infra", reason="no drift report")

    reasons, zs = [], []
    shift = fd.get("mean_shift_sigma")
    if not _num(shift):
        sm, tm, ts = fd.get("serve_mean"), fd.get("train_mean"), fd.get("train_std")
        if _num(sm) and _num(tm) and _num(ts) and ts > 0:
            shift = abs(sm - tm) / ts
    key = "shift:" + str(payload.get("feature_view", "_"))
    if _num(shift):
        bound = ctx.baseline.get("feature_mean_shift_sigma_max")
        if _num(bound) and shift > bound:
            reasons.append(f"mean_shift_sigma={shift:.3f}>max")
        else:
            # pooled history across views + per-view history; either can fire
            for name in ("mean_shift_sigma", key):
                z = _robust_z(ctx, name, shift, bound if _num(bound) else None)
                if z is not None and z > Z_ONLINE:
                    reasons.append(f"{name} robust_z={z:.2f}")
                    break
    return _verdict(ctx, "ai_infra", reasons, zs, {"mean_shift_sigma": shift, key: shift})


def check_embedding_batch(payload, ctx):
    ed = ctx.tools.embedding_drift(payload.get("corpus"), payload.get("chunk_batch_id"))
    if _bad(ed):
        return Verdict(alert=False, pillar="ai_infra", reason="no drift report")

    reasons, zs = [], []
    _upper_check(ctx, "centroid_shift", ed.get("centroid_shift"),
                 "embedding_centroid_shift_max", reasons, zs)
    _upper_check(ctx, "avg_doc_age_days", ed.get("avg_doc_age_days"),
                 "corpus_avg_doc_age_days_max", reasons, zs)
    return _verdict(ctx, "ai_infra", reasons, zs,
                    {"centroid_shift": ed.get("centroid_shift"),
                     "avg_doc_age_days": ed.get("avg_doc_age_days")})
