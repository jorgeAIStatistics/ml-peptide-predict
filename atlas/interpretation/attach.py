"""Attach interpretations to ATLAS results, using each result's real metadata.

The context handed to the interpretation engine is assembled from what actually
produced the number: the model card for a prediction, the label provenance for
a simulation, the calibration report for an interval. Nothing is supplied as a
constant where the system holds the measured value.
"""

from __future__ import annotations

import logging
from typing import Any

from atlas.interpretation.engine import concentration_vs_affinity, interpret

logger = logging.getLogger(__name__)

__all__ = ["interpret_affinity", "interpret_simulation", "interpret_model_card"]


def _applicability_phrase(similarity: float | None, band: str | None) -> str | None:
    if similarity is None:
        return None
    if similarity >= 0.7:
        strength = "well inside the domain the model was fitted on"
    elif similarity >= 0.5:
        strength = "near the edge of the fitted domain"
    else:
        strength = "at the margin of the fitted domain"
    return (
        f"Nearest training molecule at Tanimoto {similarity:.3f}"
        + (f" (band {band})" if band else "")
        + f" — {strength}."
    )


def interpret_affinity(payload: dict[str, Any], row: dict[str, Any] | None = None
                       ) -> list[dict[str, Any]]:
    """Interpret a measurement or prediction from ``/api/predict``."""
    out: list[dict[str, Any]] = []
    kind = payload.get("answer_kind")
    mw = (row or {}).get("molecular_weight")
    target = payload.get("target")

    if kind == "measurement":
        m = payload["measurement"]
        drivers = None
        if m.get("is_discordant"):
            drivers = (
                f"Reported values disagree: records span pKi {m['pki_min']:.2f} "
                f"to {m['pki_max']:.2f}, a "
                f"{10 ** (m['pki_max'] - m['pki_min']):.0f}-fold range in Ki. "
                "The dominant uncertainty is between-study disagreement, not "
                "model error."
            )
        i = interpret("pki", m["pki"], {
            "target": target, "species": "human",
            "system": "radioligand binding assay",
            "source_kind": "direct experimental measurement",
            "interval_low": m.get("pki_min"), "interval_high": m.get("pki_max"),
            "dominant_uncertainty": drivers,
            "contributions": [{
                "kind": "measurement",
                "detail": (
                    f"Median of {m.get('n_replicates')} activity records across "
                    f"{m.get('n_documents')} independent studies, from "
                    f"{m.get('molecule_chembl_id')}."
                ),
            }],
            "assumptions": [
                "Binding assay conditions differ between studies; pooled "
                "values inherit that heterogeneity.",
            ],
        })
        if i:
            out.append(i.as_dict())

        ki_nm = m.get("ki_nm")
        if ki_nm:
            j = interpret("ki", ki_nm, {"target": target,
                                        "source_kind": "derived from measured pKi"})
            if j:
                out.append(j.as_dict())

    elif kind == "prediction":
        p = payload["prediction"]
        cal = payload.get("calibration") or {}
        i = interpret("pki", p["predicted_pki"], {
            "target": target, "species": "human",
            "system": "model prediction, not an assay",
            "source_kind": "computational inference",
            "interval_low": p.get("interval_low"),
            "interval_high": p.get("interval_high"),
            "interval_nominal": p.get("interval_nominal"),
            "interval_measured_coverage": p.get("interval_empirical_coverage"),
            "applicability": _applicability_phrase(
                payload.get("max_similarity"), payload.get("similarity_band")),
            "dominant_uncertainty": (
                "Model error on unseen chemotypes. The interval comes from "
                "conformal calibration on a scaffold-disjoint block, and its "
                "width reflects how imprecise the model is there rather than "
                "any property of this molecule."
            ),
            "model_performance": cal.get("empirical_coverage") and {
                "empirical_coverage": cal.get("empirical_coverage"),
                "nominal_coverage": cal.get("nominal_coverage"),
                "median_interval_width": cal.get("median_interval_width_log"),
            },
            "contributions": [
                {
                    "kind": "nearest_training_molecule",
                    "detail": (
                        f"{n['molecule_chembl_id']} at Tanimoto "
                        f"{n['tanimoto']:.3f}, measured pKi {n['measured_pki']:.2f}"
                    ),
                }
                for n in p.get("nearest_neighbours", [])
            ] + [{
                "kind": "model",
                "detail": (
                    "A similarity-weighted nearest-neighbour regressor: the "
                    "prediction is an average over those measured molecules, "
                    "which is the whole of its reasoning."
                ),
            }],
            "assumptions": [
                "That fingerprint similarity tracks activity similarity, which "
                "activity cliffs violate.",
                "That the target relevance established for this molecule holds "
                "under physiological conditions.",
            ],
        })
        if i:
            out.append(i.as_dict())

    # Cross-endpoint: relate exposure to affinity when both are known.
    pki = None
    if kind == "measurement":
        pki = payload["measurement"]["pki"]
    elif kind == "prediction":
        pki = payload["prediction"]["predicted_pki"]
    cmax = (row or {}).get("label_cmax_ng_per_ml")
    if pki is not None and cmax and mw:
        bridge = concentration_vs_affinity(
            cmax, pki, mw, unbound_fraction=(row or {}).get("unbound_fraction"))
        out.append({
            "endpoint": "exposure_vs_affinity",
            "symbol": "Cmax/Ki",
            "name": "Peak exposure relative to binding affinity",
            "value": bridge["ratio_to_ki"], "unit": "ratio",
            "what_is_measured": {
                "definition": (
                    "The ratio of peak plasma concentration to the binding "
                    "affinity, which indicates whether exposure reaches the "
                    "concentration range where the target would be engaged."
                ),
                "target": target, "source_kind": "derived from two measurements",
            },
            "mathematics": {
                "transformation": "ratio = C_free/Ki; occupancy = C/(C + Ki)",
                "stated_value": bridge["ratio_phrase"],
            },
            "derived": [
                {"label": "Peak concentration", "value": bridge["peak_concentration"],
                 "how": "Cmax converted to molar using the molecular weight"},
                {"label": "Unbound peak", "value": bridge["unbound_peak_concentration"]
                 or "not computed (no protein binding value)", "how": "free fraction only"},
                {"label": "Binding affinity", "value": bridge["ki"], "how": "from pKi"},
                {"label": "Implied occupancy at peak",
                 "value": f"{bridge['implied_occupancy_at_peak'] * 100:.0f}%",
                 "how": "equilibrium occupancy model"},
            ],
            "biological_meaning": {
                "establishes": [
                    "Whether systemic exposure reaches the concentration range "
                    "at which the target binds.",
                ],
                "does_not_establish": [
                    "A functional response at that occupancy.",
                    "Concentration at the target tissue.",
                ],
                "common_misreadings": bridge["caveats"],
            },
            "uncertainty": {"drivers": bridge["caveats"]},
            "why_this_result": [{
                "kind": "composition",
                "detail": "Computed from a measured Cmax and a measured Ki; no "
                          "model is involved in the ratio itself.",
            }],
            "validation": {
                "measurements_that_would_test_it": [
                    "Receptor occupancy imaging at the dose of interest.",
                    "A functional pharmacodynamic endpoint measured against "
                    "concentration.",
                ],
            },
        })

    return out


def interpret_simulation(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Interpret the derived quantities of a ``/api/simulate`` run."""
    out: list[dict[str, Any]] = []
    nca = payload.get("nca") or {}
    diag = payload.get("diagnostics") or {}
    params = payload.get("parameters") or {}
    is_exploratory = str(payload.get("evidence_class", "")).startswith("D")

    source_kind = (
        "exploratory simulation on user-supplied parameters"
        if is_exploratory else "simulation on source-backed parameters"
    )
    assumptions = [
        f"A {payload.get('model_display_name', 'compartmental')} structure was "
        "assumed; a different structure gives different peak behaviour.",
        "Linear kinetics: clearance and volume do not vary with concentration.",
    ]
    if is_exploratory:
        assumptions.insert(0, (
            "The parameters were supplied by the user and are not tied to any "
            "measurement, so these quantities describe the assumptions rather "
            "than a compound."
        ))

    contributions = [
        {"kind": "parameter", "detail": f"{k} = {v:g}"} for k, v in params.items()
    ] + [{
        "kind": "solver",
        "detail": (
            f"{diag.get('solver', {}).get('method', 'LSODA')} at rtol "
            f"{diag.get('solver', {}).get('rtol')}, mass balance conserved to "
            f"{diag.get('mass_balance_relative_error', 0):.1e} relative."
        ),
    }]

    cmax = nca.get("cmax_mg_per_l")
    if cmax is not None:
        i = interpret("cmax", cmax * 1000.0, {
            "unit": "ng/mL", "source_kind": source_kind,
            "contributions": contributions, "assumptions": assumptions,
            "dominant_uncertainty": (
                "Peak height is the quantity most sensitive to model structure "
                "and to the absorption rate constant, which is commonly derived "
                "from a reported Tmax rather than measured."
            ),
        })
        if i:
            out.append(i.as_dict())

    tmax = nca.get("tmax_h")
    if tmax is not None:
        i = interpret("tmax", tmax, {
            "source_kind": source_kind, "contributions": contributions,
            "assumptions": assumptions,
        })
        if i:
            out.append(i.as_dict())

    auc = nca.get("analytic_auc_inf_mg_h_per_l")
    if auc is not None:
        extrap = nca.get("extrapolated_fraction")
        i = interpret("auc", auc * 1000.0, {
            "unit": "ng·h/mL", "source_kind": source_kind,
            "dose_mg": payload.get("dose_mg"),
            "bioavailability": params.get("F"),
            "contributions": contributions, "assumptions": assumptions,
            "dominant_uncertainty": (
                f"{extrap * 100:.1f}% of the numerically integrated AUC came "
                "from extrapolation beyond the last sample."
                if extrap else None
            ),
        })
        if i:
            out.append(i.as_dict())

    half = nca.get("terminal_half_life_h") or diag.get("terminal_half_life_h")
    if half is not None:
        ctx = {
            "source_kind": source_kind, "contributions": contributions,
            "assumptions": assumptions,
            "dosing_interval_h": payload.get("interval_h"),
        }
        if diag.get("is_flip_flop"):
            ctx["dominant_uncertainty"] = (
                "Absorption is slower than elimination, so the terminal slope "
                "reflects absorption. Reading clearance from this phase would "
                "give the wrong answer."
            )
        i = interpret("half_life", half, ctx)
        if i:
            out.append(i.as_dict())

    return out


def interpret_model_card(card: dict[str, Any], calibration: dict[str, Any],
                         noise: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Interpret the headline performance numbers of the deployed model."""
    out: list[dict[str, Any]] = []
    hp = card.get("headline_performance") or {}
    rs = card.get("random_split_comparison") or {}

    mae = hp.get("mae_log_units")
    if mae is not None:
        i = interpret("mae", mae, {
            "source_kind": f"held-out evaluation, {hp.get('split')}",
            "experimental_noise_floor": (noise or {}).get("implied_mae_floor_log"),
            "dominant_uncertainty": (
                "Evaluation-set construction. The same model scores "
                f"{rs.get('mae_log_units')} on a random split, where "
                f"{(rs.get('fraction_test_scaffolds_seen_in_training') or 0) * 100:.0f}% "
                "of test scaffolds were already seen in training."
            ),
            "contributions": [{
                "kind": "evaluation",
                "detail": f"Measured on {hp.get('split')}.",
            }],
        })
        if i:
            out.append(i.as_dict())

    r2 = hp.get("r2")
    if r2 is not None:
        i = interpret("r2", r2, {"source_kind": f"held-out, {hp.get('split')}"})
        if i:
            out.append(i.as_dict())

    cov = calibration.get("empirical_coverage")
    if cov is not None:
        i = interpret("coverage", cov, {
            "nominal_coverage": calibration.get("nominal_coverage"),
            "source_kind": calibration.get("method"),
            "dominant_uncertainty": (
                "Calibration previously measured "
                f"{calibration.get('previous_empirical_coverage')} when the "
                "calibration set was drawn from the training pool rather than "
                "from a scaffold-disjoint block."
            ),
            "contributions": [{
                "kind": "calibration",
                "detail": f"{calibration.get('data_roles')}.",
            }],
        })
        if i:
            out.append(i.as_dict())

    return out
