"""JSON API over the ATLAS backend.

Reads from the versioned snapshots written by ``atlas.registry.build`` and
``atlas.analysis.report`` rather than querying external databases per request:
the API is therefore fast, reproducible and usable offline, and every response
carries the ``snapshot_id`` its evidence came from.

Simulation is the one live computation. It runs the real solver, and its result
is labelled by the evidence gate: unless source-backed pharmacokinetic
parameters exist for a compound, a run is returned as class D (exploratory)
with the abstention attached, never as a statement about humans.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

from atlas.analysis.feasibility import AnalysisKind
from atlas.core.evidence import AbstentionReason, EvidenceClass
from atlas.mechanistic.metrics import nca
from atlas.mechanistic.models import MODEL_REGISTRY, ParameterError, get_model
from atlas.mechanistic.solver import DoseEvent, SolverError, SolverSettings, simulate
from atlas.ml.predict import MODEL_CARD, get_predictor
from atlas.ml.peptide_service import PEPTIDE_MODEL_FINDINGS
from atlas.ml.affinity_service import CALIBRATION_CARD, get_affinity_service
from atlas.registry.seed import by_key as seed_by_key
from atlas.mechanistic.pk_store import load_store, parameters_for
from atlas.interpretation.attach import (
    interpret_affinity, interpret_model_card, interpret_simulation,
)
from atlas.registry.seed import RESEARCH_AREAS, SEED_REGISTRY

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STATIC = Path(__file__).resolve().parent / "static"

#: Hard cap on simulation output points, so a request cannot exhaust memory.
MAX_OUTPUT_POINTS = 4000
MAX_DOSES = 200


# ---------------------------------------------------------------------------
# snapshot loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_registry_snapshot() -> dict[str, Any]:
    path = DATA / "registry_snapshot.json"
    if not path.exists():
        return {"snapshot_id": None, "n_entries": 0, "identities": []}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_feasibility_report() -> dict[str, Any]:
    path = DATA / "feasibility_report.json"
    if not path.exists():
        return {"created_at": None, "n_entries": 0, "rows": []}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def seed_index() -> dict[str, Any]:
    return {
        e.key: {
            "key": e.key,
            "name": e.name,
            "research_area": e.research_area,
            "declared_modality": e.declared_modality.value,
            "aliases": list(e.aliases),
            "is_mixture": e.is_mixture,
            "supplier_defined": e.supplier_defined,
            "reported_components": list(e.reported_components),
            "identity_note": e.identity_note,
        }
        for e in SEED_REGISTRY
    }


@lru_cache(maxsize=1)
def merged_rows() -> list[dict[str, Any]]:
    """Join seed facts, resolved identity and measured feasibility by key."""
    identities = {i["registry_key"]: i for i in load_registry_snapshot()["identities"]}
    feasibility = {r["key"]: r for r in load_feasibility_report()["rows"]}
    seeds = seed_index()

    rows: list[dict[str, Any]] = []
    for key, seed in seeds.items():
        ident = identities.get(key, {})
        feas = feasibility.get(key, {})
        rows.append(
            {
                **seed,
                "identity_status": ident.get("status"),
                "modality": ident.get("modality", seed["declared_modality"]),
                "preferred_name": (ident.get("preferred_name") or {}).get("value"),
                "chembl_id": (ident.get("chembl_id") or {}).get("value"),
                "pubchem_cid": (ident.get("pubchem_cid") or {}).get("value"),
                "molecular_weight": (
                    ident.get("molecular_weight_g_per_mol") or {}
                ).get("value"),
                "molecular_formula": (ident.get("molecular_formula") or {}).get("value"),
                "smiles": (ident.get("smiles") or {}).get("value"),
                "max_clinical_phase": (ident.get("max_clinical_phase") or {}).get("value"),
                "caveats": ident.get("caveats", []),
                "completeness": feas.get("completeness"),
                "approved": feas.get("approved", False),
                "publications": feas.get("publications"),
                "human_publications": feas.get("human_publications"),
                "human_pk_publications": feas.get("human_pk_publications"),
                "randomized_trials": feas.get("randomized_trials"),
                "registered_trials": feas.get("registered_trials"),
                "trials_with_results": feas.get("trials_with_results"),
                "density": feas.get("density"),
                "permitted": feas.get("permitted", []),
                "refused": feas.get("refused", {}),
            }
        )
    rows.sort(key=lambda r: -(r["publications"] or 0))
    return rows


@lru_cache(maxsize=1)
def ml_report() -> dict[str, Any]:
    """The model registry written by ``atlas.ml.train``."""
    path = DATA / "ml" / "model_registry.json"
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    payload["available"] = True
    return payload


@lru_cache(maxsize=1)
def peptide_report() -> dict[str, Any]:
    """The peptide experiment registry written by ``atlas.ml.peptide_train``."""
    path = DATA / "ml" / "peptide_model_registry.json"
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    payload["available"] = True
    return payload


def _json_file(name: str) -> dict[str, Any]:
    path = DATA / "ml" / name
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text())
    payload["available"] = True
    return payload


@lru_cache(maxsize=1)
def calibration_report() -> dict[str, Any]:
    return _json_file("calibration_report.json")


@lru_cache(maxsize=1)
def peptide_v2_report() -> dict[str, Any]:
    return _json_file("peptide_v2_report.json")


@lru_cache(maxsize=1)
def verification_payload() -> dict[str, Any]:
    """Run the solver verification suite once and cache it in memory."""
    from atlas.mechanistic.verify import verify_all

    reports = verify_all()
    out = []
    for r in reports:
        out.append(
            {
                "model": r.model_name,
                "passed": r.passed,
                "n_passed": r.n_passed,
                "n_checks": len(r.checks),
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "metric": None if np.isnan(c.metric) else c.metric,
                        "tolerance": c.tolerance,
                        "units": c.units,
                        "detail": c.detail,
                    }
                    for c in r.checks
                ],
            }
        )
    return {
        "reports": out,
        "n_checks": sum(r["n_checks"] for r in out),
        "n_passed": sum(r["n_passed"] for r in out),
        "all_passed": all(r["passed"] for r in out),
        "note": (
            "These checks establish numerical correctness only: that the solver "
            "solves the equations it states, against closed-form solutions and "
            "conservation laws. They are not evidence that any model describes a "
            "real biological system."
        ),
    }


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
def _finite_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _finite_json(obj.item())
    if isinstance(obj, dict):
        return {k: _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


class NumpyJSONProvider(DefaultJSONProvider):
    """Serialise numpy scalars and arrays.

    Scientific code returns numpy scalars from arithmetic on arrays. Rather
    than casting at every call site and depending on none being missed, the
    boundary handles them.
    """

    #: Bare NaN and Infinity are not valid JSON and browsers reject them.
    #: Undefined statistics are emitted as null instead.
    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("allow_nan", False)
        return super().dumps(_finite_json(obj), **kwargs)

    @staticmethod
    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json = NumpyJSONProvider(app)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.get("/api/meta")
    def meta():
        snap = load_registry_snapshot()
        feas = load_feasibility_report()
        rows = merged_rows()
        return jsonify(
            {
                "snapshot_id": snap.get("snapshot_id"),
                "registry_created_at": snap.get("created_at"),
                "feasibility_created_at": feas.get("created_at"),
                "n_entries": len(rows),
                "research_areas": list(RESEARCH_AREAS),
                "analysis_kinds": [
                    {"value": k.value, "label": k.label} for k in AnalysisKind
                ],
                "evidence_classes": [
                    {"value": c.value, "label": c.label, "rank": c.rank}
                    for c in EvidenceClass
                ],
                "identity_status_counts": _count(rows, "identity_status"),
                "density_counts": _count(rows, "density"),
                "permitted_counts": {
                    k.value: sum(1 for r in rows if k.value in r["permitted"])
                    for k in AnalysisKind
                },
                "scope_note": (
                    "Research platform for literature analysis and computational "
                    "modelling. It does not provide dosing, administration or "
                    "combination guidance, and produces no individual risk estimates."
                ),
            }
        )

    @app.get("/api/registry")
    def registry():
        rows = merged_rows()
        area = request.args.get("area")
        status = request.args.get("identity_status")
        query = (request.args.get("q") or "").strip().lower()
        if area:
            rows = [r for r in rows if r["research_area"] == area]
        if status:
            rows = [r for r in rows if r["identity_status"] == status]
        if query:
            rows = [
                r
                for r in rows
                if query in r["name"].lower()
                or any(query in a.lower() for a in r["aliases"])
                or query in (r["preferred_name"] or "").lower()
            ]
        return jsonify({"n": len(rows), "rows": rows})

    @app.get("/api/compound/<key>")
    def compound(key: str):
        row = next((r for r in merged_rows() if r["key"] == key), None)
        if row is None:
            return jsonify({"error": f"no registry entry {key!r}"}), 404
        identities = {
            i["registry_key"]: i for i in load_registry_snapshot()["identities"]
        }
        ident = identities.get(key, {})
        # The flattened row carries values; ``identity_fields`` carries each
        # field's own source, so the inspector can attribute every number
        # rather than implying one database supplied the whole identity.
        field_names = (
            "preferred_name", "chembl_id", "pubchem_cid",
            "molecular_weight_g_per_mol", "molecular_formula", "smiles",
            "inchikey", "molecule_type", "max_clinical_phase",
            "first_approval_year",
        )
        return jsonify(
            {
                **row,
                "identity_fields": {
                    n: ident.get(n) for n in field_names if ident.get(n)
                },
                "sources": ident.get("sources", []),
                "synonyms": ident.get("synonyms", []),
                "candidates_considered": ident.get("candidates_considered", []),
                "snapshot_id": load_registry_snapshot().get("snapshot_id"),
            }
        )

    @app.get("/api/models")
    def models():
        out = []
        for name, cls in MODEL_REGISTRY.items():
            m = cls()
            out.append(
                {
                    "name": name,
                    "display_name": getattr(m, "display_name", name),
                    "description": m.description,
                    "equations": list(m.equations),
                    "state_names": list(m.state_names),
                    "dose_compartment": (
                        "depot_mg" if "depot_mg" in m.state_names else "central_mg"
                    ),
                    "parameters": [
                        {
                            "name": p.name,
                            "unit": p.unit,
                            "description": p.description,
                            "lower": p.lower,
                            "upper": p.upper,
                            "dimension": str(p.dimension),
                        }
                        for p in m.parameters
                    ],
                }
            )
        return jsonify({"models": out})

    @app.get("/api/verification")
    def verification():
        return jsonify(verification_payload())

    @app.get("/api/model-card")
    def model_card():
        """The deployed model's card plus the full training comparison."""
        return jsonify({
            "card": MODEL_CARD,
            "report": ml_report(),
            "peptide_findings": PEPTIDE_MODEL_FINDINGS,
            "peptide_report": peptide_report(),
            "calibration": CALIBRATION_CARD,
            "calibration_report": calibration_report(),
            "peptide_v2_report": peptide_v2_report(),
            "interpretations": interpret_model_card(
                MODEL_CARD, CALIBRATION_CARD,
                (ml_report() or {}).get("noise_estimate"),
            ),
        })

    @app.post("/api/predict")
    def predict():
        """Affinity for a molecule, gated on evidence at every step.

        Returns a measurement where one exists, a calibrated prediction only
        where target relevance and applicability both hold, and otherwise a
        structured refusal. A model-generated number is never returned for a
        compound with no established relationship to the target.
        """
        body = request.get_json(silent=True) or {}
        smiles = body.get("smiles")
        key = body.get("compound_key")
        compound_name = None
        names: tuple[str, ...] = ()

        if not smiles and key:
            row = next((r for r in merged_rows() if r["key"] == key), None)
            if row is None:
                return jsonify({"error": f"no registry entry {key!r}"}), 404
            compound_name = row["name"]
            try:
                entry = seed_by_key(key)
                names = (entry.name, *entry.aliases)
            except KeyError:
                names = (row["name"],)
            smiles = row.get("smiles")
            if not smiles:
                return jsonify({
                    "answer_kind": "abstention",
                    "smiles": None, "compound_key": key,
                    "compound_name": compound_name,
                    "evidence_class": EvidenceClass.INSUFFICIENT.value,
                    "target_relevance": "D_unsupported_target_relationship",
                    "target_relevance_label": "Unsupported target relationship",
                    "may_create_graph_edge": False,
                    "abstention": {
                        "reason": (
                            AbstentionReason.UNRESOLVED_MOLECULAR_IDENTITY.value
                        ),
                        "explanation": (
                            f"No structure is resolved for {compound_name}, so it "
                            "can neither be matched against measured data nor "
                            f"featurised. Identity status: {row.get('identity_status')}."
                        ),
                        "what_would_resolve_it": [
                            "A SMILES string, InChI or sequence for the intended "
                            "molecular form.",
                        ],
                    },
                })

        if not smiles:
            return jsonify({"error": "supply 'smiles' or 'compound_key'"}), 400
        if len(smiles) > 5000:
            return jsonify({"error": "SMILES string is too long"}), 400

        try:
            r = get_affinity_service().answer(smiles, names=names)
        except Exception as exc:
            logger.exception("affinity lookup failed")
            return jsonify({"error": f"lookup failed: {exc}"}), 500

        payload: dict[str, Any] = {
            "answer_kind": r.answer_kind,
            "smiles": smiles,
            "compound_key": key,
            "compound_name": compound_name,
            "evidence_class": r.evidence_class,
            "target": r.target,
            "modality": r.modality,
            "target_relevance": r.target_relevance,
            "target_relevance_label": r.target_relevance_label,
            "target_rationale": r.target_rationale,
            "target_evidence": r.target_evidence,
            "may_create_graph_edge": r.may_create_graph_edge,
            "max_similarity": r.max_similarity,
            "similarity_band": r.similarity_band,
            "limitations": r.limitations,
            "dataset_version": r.dataset_version,
        }

        if r.answer_kind == "measurement":
            payload["measurement"] = {
                "pki": r.pki, "pki_min": r.pki_min, "pki_max": r.pki_max,
                "ki_nm": r.ki_nm, "is_discordant": r.is_discordant,
                "molecule_chembl_id": r.molecule_chembl_id,
                "chembl_url": (
                    "https://www.ebi.ac.uk/chembl/compound_report_card/"
                    f"{r.molecule_chembl_id}/"
                ),
                "n_replicates": r.n_replicates,
                "n_documents": r.n_documents,
            }
        elif r.answer_kind == "prediction":
            payload["prediction"] = {
                "predicted_pki": r.predicted_pki,
                "predicted_ki_nm": (
                    None if r.predicted_pki is None
                    else float(10 ** (9.0 - r.predicted_pki))
                ),
                "interval_low": r.interval_low,
                "interval_high": r.interval_high,
                "interval_nominal": r.interval_nominal,
                "interval_empirical_coverage": r.interval_empirical_coverage,
                "nearest_neighbours": r.nearest_neighbours,
            }
            payload["calibration"] = r.calibration
        else:
            payload["abstention"] = {
                "reason": r.abstention_reason,
                "explanation": r.abstention_explanation,
                "what_would_resolve_it": r.what_would_resolve_it,
            }

        # Every reported number carries its interpretation, built from the
        # metadata of whatever produced it.
        row = next((x for x in merged_rows() if x["key"] == key), None) if key else None
        if row is not None:
            pk = parameters_for(key) or {}
            extracted = {e["name"]: e for e in pk.get("extracted", [])}
            if "cmax" in extracted:
                row = dict(row, label_cmax_ng_per_ml=extracted["cmax"]["value"])
            if "protein_binding" in extracted:
                row = dict(row,
                           unbound_fraction=1.0 - extracted["protein_binding"]["value"])
        try:
            payload["interpretations"] = interpret_affinity(payload, row)
        except Exception:
            logger.exception("interpretation failed")
            payload["interpretations"] = []
        return jsonify(payload)

    @app.get("/api/pk-parameters")
    def pk_parameters():
        """Source-backed PK parameters and their validation against the label."""
        store = load_store()
        return jsonify({
            "n_attempted": store.get("n_attempted", 0),
            "n_with_label": store.get("n_with_label", 0),
            "n_usable": store.get("n_usable", 0),
            "usable_keys": store.get("usable_keys", []),
            "note": store.get("note"),
            "entries": store.get("entries", {}),
        })

    @app.get("/api/calibration")
    def calibration():
        return jsonify({
            "card": CALIBRATION_CARD,
            "report": calibration_report(),
        })

    @app.post("/api/simulate")
    def run_simulation():
        body = request.get_json(silent=True) or {}
        model_name = body.get("model")
        params = body.get("parameters") or {}
        dose_mg = body.get("dose_mg")
        n_doses = int(body.get("n_doses", 1))
        interval_h = body.get("interval_h")
        horizon_h = body.get("horizon_h")
        n_points = int(body.get("n_points", 600))
        compound_key = body.get("compound_key")

        try:
            model = get_model(model_name) if model_name else None
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 400
        if model is None:
            return jsonify({"error": "field 'model' is required"}), 400

        if not isinstance(dose_mg, (int, float)) or dose_mg <= 0:
            return jsonify({"error": "'dose_mg' must be a positive number"}), 400
        if not 1 <= n_doses <= MAX_DOSES:
            return jsonify({"error": f"'n_doses' must be 1..{MAX_DOSES}"}), 400
        if not 2 <= n_points <= MAX_OUTPUT_POINTS:
            return jsonify({"error": f"'n_points' must be 2..{MAX_OUTPUT_POINTS}"}), 400

        try:
            clean = model.validate({k: float(v) for k, v in params.items()})
        except (ParameterError, TypeError, ValueError) as exc:
            # Real parameter validation errors are surfaced verbatim: a wrong
            # unit or a missing parameter should be visible, not smoothed over.
            return jsonify({"error": str(exc), "error_kind": "parameter"}), 400

        half_life = model.terminal_half_life_h(clean) or 1.0
        if interval_h is None:
            interval_h = max(half_life, 1e-6)
        interval_h = float(interval_h)
        if interval_h <= 0:
            return jsonify({"error": "'interval_h' must be positive"}), 400
        if horizon_h is None:
            horizon_h = (n_doses - 1) * interval_h + 8.0 * half_life
        horizon_h = float(horizon_h)
        if horizon_h <= 0:
            return jsonify({"error": "'horizon_h' must be positive"}), 400

        compartment = "depot_mg" if "depot_mg" in model.state_names else "central_mg"
        doses = [
            DoseEvent(i * interval_h, float(dose_mg), compartment)
            for i in range(n_doses)
            if i * interval_h <= horizon_h
        ]
        times = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, horizon_h, n_points),
                    np.array([d.time_h for d in doses]),
                ]
            )
        )

        try:
            res = simulate(model, clean, doses, times, settings=SolverSettings())
        except (SolverError, ValueError, ParameterError) as exc:
            return jsonify({"error": str(exc), "error_kind": "solver"}), 400

        conc = res.concentration_mg_per_l
        analysis = nca(res.times_h, conc)
        exact_auc = model.analytical_auc_inf(float(dose_mg) * len(doses), clean)

        label, abstention = _label_simulation(compound_key)

        return jsonify(
            {
                "run_id": res.run_id,
                "model": model.name,
                "model_display_name": getattr(model, "display_name", model.name),
                "equations": list(model.equations),
                "parameters": clean,
                "dose_mg": dose_mg,
                "n_doses": len(doses),
                "interval_h": interval_h,
                "horizon_h": horizon_h,
                "times_h": [round(float(t), 6) for t in res.times_h],
                "concentration_mg_per_l": [float(c) for c in conc],
                "states": {
                    name: [float(v) for v in res.states_mg[i]]
                    for i, name in enumerate(res.state_names)
                },
                "diagnostics": {
                    "mass_balance_max_abs_error_mg": res.mass_balance_max_abs_error_mg,
                    "mass_balance_relative_error": res.mass_balance_relative_error,
                    "solver": res.settings.as_dict(),
                    "terminal_half_life_h": half_life,
                    "is_flip_flop": bool(
                        getattr(model, "is_flip_flop", lambda _p: False)(clean)
                    ),
                },
                "nca": {
                    "cmax_mg_per_l": analysis.cmax_mg_per_l,
                    "tmax_h": analysis.tmax_h,
                    "auc_last_mg_h_per_l": analysis.auc_last_mg_h_per_l,
                    "auc_inf_mg_h_per_l": analysis.auc_inf_mg_h_per_l,
                    "extrapolated_fraction": analysis.extrapolated_fraction,
                    "auc_inf_is_reliable": analysis.auc_inf_is_reliable,
                    "terminal_half_life_h": analysis.terminal.half_life_h,
                    "terminal_r_squared": analysis.terminal.r_squared,
                    "terminal_n_points": analysis.terminal.n_points,
                    "terminal_reason": analysis.terminal.reason,
                    "analytic_auc_inf_mg_h_per_l": exact_auc,
                },
                "evidence_class": label["evidence_class"],
                "evidence_label": label["evidence_label"],
                "interpretation": label["interpretation"],
                "abstention": abstention,
                "compound_key": compound_key,
                "interpretations": _simulation_interpretations(
                    model, clean, dose_mg, doses, interval_h, res, analysis,
                    exact_auc, half_life, label,
                ),
            }
        )

    # -- static frontend ---------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(STATIC, filename)

    return app


def _simulation_interpretations(model, params, dose_mg, doses, interval_h,
                                res, analysis, exact_auc, half_life, label):
    """Build interpretations for a simulation's derived quantities."""
    try:
        return interpret_simulation({
            "evidence_class": label["evidence_class"],
            "model_display_name": getattr(model, "display_name", model.name),
            "parameters": params,
            "dose_mg": dose_mg,
            "interval_h": interval_h if len(doses) > 1 else None,
            "diagnostics": {
                "mass_balance_relative_error": res.mass_balance_relative_error,
                "solver": res.settings.as_dict(),
                "terminal_half_life_h": half_life,
                "is_flip_flop": bool(
                    getattr(model, "is_flip_flop", lambda _p: False)(params)),
            },
            "nca": {
                "cmax_mg_per_l": analysis.cmax_mg_per_l,
                "tmax_h": analysis.tmax_h,
                "analytic_auc_inf_mg_h_per_l": exact_auc,
                "extrapolated_fraction": analysis.extrapolated_fraction,
                "terminal_half_life_h": analysis.terminal.half_life_h,
            },
        })
    except Exception:
        logger.exception("simulation interpretation failed")
        return []


def _count(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        key = r.get(field) or "unknown"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _label_simulation(compound_key: str | None) -> tuple[dict[str, str], dict | None]:
    """Decide how a simulation result may be presented.

    A solver will integrate whatever parameters it is given. Whether the output
    may be described as the pharmacokinetics of a named compound depends on the
    evidence gate, not on the mathematics.
    """
    exploratory = {
        "evidence_class": EvidenceClass.EXPLORATORY_HYPOTHESIS.value,
        "evidence_label": EvidenceClass.EXPLORATORY_HYPOTHESIS.label,
        "interpretation": (
            "Exploratory mathematical run on user-supplied parameters. It shows "
            "the behaviour of these equations under these assumptions and is "
            "not a prediction about any organism."
        ),
    }
    if not compound_key:
        return exploratory, None

    row = next((r for r in merged_rows() if r["key"] == compound_key), None)
    if row is None:
        return exploratory, None

    pk = parameters_for(compound_key)
    if pk and AnalysisKind.HUMAN_PK_SIMULATION.value in row["permitted"]:
        v = pk.get("validation") or {}
        return (
            {
                "evidence_class": EvidenceClass.REPRODUCED_MODEL.value,
                "evidence_label": EvidenceClass.REPRODUCED_MODEL.label,
                "interpretation": (
                    "Parameters come from the FDA label for "
                    f"{pk.get('brand_name') or pk.get('generic_name')}, each with "
                    "the sentence that states it. Against the quantities that "
                    f"label reports independently, the model {v.get('verdict', '')}."
                ),
            },
            None,
        )

    reason = row["refused"].get(AnalysisKind.HUMAN_PK_SIMULATION.value)
    abstention = {
        "reason": reason,
        "endpoint": f"human pharmacokinetics for {row['name']}",
        "explanation": (
            f"This run is not a statement about {row['name']} in humans. The "
            "evidence gate refused a human pharmacokinetic simulation for this "
            f"compound ({reason}). Measured evidence: "
            f"{row.get('human_pk_publications')} human PK publications, "
            f"{row.get('trials_with_results')} trials with posted results."
        ),
        "what_would_resolve_it": [
            "Clearance, volume of distribution and bioavailability extracted "
            "from the identified records, with units and uncertainty, verified "
            "against the source text.",
        ],
    }
    return exploratory, abstention


app = create_app()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the ATLAS API and interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
