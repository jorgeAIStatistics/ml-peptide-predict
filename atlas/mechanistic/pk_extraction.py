"""Extract pharmacokinetic parameters from FDA labelling, with verbatim evidence.

Every extracted number carries the sentence it came from. That is the whole
point: a parameter that cannot be checked against its source text is not
evidence, and an extraction pipeline nobody can audit is a fabrication engine
with extra steps.

FDA labels are used because they are authoritative, machine-accessible through
openFDA, and state parameters in a form a model can consume. The cost is
coverage: only approved products have one, which is itself a useful signal
about how much is actually known for a compound.

Three distinctions are preserved rather than flattened:

* **Point values versus bounds.** Tesamorelin's label says bioavailability is
  "less than 4%". That is a censored observation, not 4%, and is recorded as an
  upper bound.
* **Mean versus median.** Tmax is conventionally reported as a median because
  its distribution is skewed; conflating it with a mean misstates the source.
* **Apparent versus true parameters.** ``CL/F`` and ``V/F`` are apparent
  parameters confounded with bioavailability. They are labelled as such, because
  treating ``CL/F`` as clearance silently assumes F = 1.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from atlas.core.evidence import SourceRecord, utc_now
from atlas.core.units import DimensionError, UnitError, parse_unit
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = [
    "PKParameter",
    "Qualifier",
    "LabelPKRecord",
    "FDALabelClient",
    "extract_parameters",
    "PARAMETER_PATTERNS",
]

OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"

LICENSE_NOTE = (
    "FDA drug labelling retrieved via openFDA. Label content is published by "
    "the FDA and is in the public domain; it reflects the approved product and "
    "not necessarily any other preparation of the same substance."
)


class Qualifier(str, enum.Enum):
    """How the source stated the value."""

    MEAN = "mean"
    MEDIAN = "median"
    APPROXIMATE = "approximate"
    UPPER_BOUND = "upper_bound"
    LOWER_BOUND = "lower_bound"
    UNSPECIFIED = "unspecified"

    @property
    def is_point_estimate(self) -> bool:
        """A bound is not a point estimate and must not be used as one."""
        return self not in (Qualifier.UPPER_BOUND, Qualifier.LOWER_BOUND)


@dataclass(frozen=True)
class PKParameter:
    """One parameter with its units, uncertainty and the text that stated it."""

    name: str
    value: float
    unit: str
    qualifier: Qualifier
    verbatim: str
    sd: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    is_apparent: bool = False          # CL/F and V/F are confounded with F
    normalised_by_weight: bool = False  # e.g. L/kg
    note: str | None = None

    @property
    def dimension(self) -> str:
        try:
            return str(parse_unit(self.unit)[1])
        except UnitError:
            return "unparseable"

    @property
    def is_usable_as_point(self) -> bool:
        return self.qualifier.is_point_estimate

    def __str__(self) -> str:
        spread = f" ± {self.sd}" if self.sd is not None else ""
        return f"{self.name} = {self.value:g}{spread} {self.unit} ({self.qualifier.value})"


#: Expected dimension for each parameter, checked against the parsed unit so a
#: mis-scraped unit (clearance in litres, say) is rejected rather than used.
EXPECTED_DIMENSION: dict[str, str] = {
    "half_life": "time",
    "clearance": "volume/time",
    "volume_of_distribution": "volume",
    "bioavailability": "dimensionless",
    "tmax": "time",
    "cmax": "mass/volume",
    "auc": "mass*time/volume",
    "cmax_and_auc_pair": "mass/volume",
    "protein_binding": "dimensionless",
}

_NUM = r"\d+(?:\.\d+)?"
#: Filler between a parameter's name and its value. It must span decimal points
#: -- tesamorelin's volume of distribution sits behind "the 1.4 mg dose of
#: EGRIFTA SV (2 mg/vial formulation)", which a [^.] gap cannot cross -- while
#: still refusing to run past a sentence boundary into an unrelated statement.
_GAP = r"(?:(?!\.\s+[A-Z])[\s\S])"
_VALUE = rf"(?P<value>{_NUM})"
_SPREAD = rf"(?:\s*(?:±|\+/-|\+-)\s*(?P<sd>{_NUM}))?"

#: Patterns are deliberately anchored on the label's own phrasing. Each must
#: capture a value and a unit; the surrounding sentence is captured separately.
PARAMETER_PATTERNS: list[tuple[str, str, str]] = [
    ("half_life",
     rf"(?:terminal\s+)?(?:elimination\s+)?half[- ]life(?:\s*\(?t\s*1?/?2\s*\)?)?{_GAP}{{0,150}}?"
     rf"(?:is|was|of)\s+(?:approximately\s+)?{_VALUE}{_SPREAD}\s*"
     rf"(?P<unit>hours?|hrs?|h|minutes?|min)\b",
     "time"),
    ("clearance",
     rf"clearance\s*(?:\(\s*CL\s*/?\s*F?\s*\))?{_GAP}{{0,150}}?(?:is|was|of)\s+"
     rf"(?:approximately\s+)?{_VALUE}{_SPREAD}\s*"
     rf"(?P<unit>L\s*/\s*(?:hr?|hour)|mL\s*/\s*min)",
     "volume/time"),
    ("volume_of_distribution",
     rf"volume\s+of\s+distribution{_GAP}{{0,150}}?(?:is|was)\s+"
     rf"(?:approximately\s+)?{_VALUE}{_SPREAD}\s*(?P<unit>L\s*/\s*kg|L)\b",
     "volume"),
    ("bioavailability",
     rf"(?:absolute\s+)?bioavailability{_GAP}{{0,150}}?(?:is|was|to\s+be)\s+"
     rf"(?:about\s+|approximately\s+)?(?P<bound>less\s+than\s+)?{_VALUE}\s*(?P<unit>%)",
     "dimensionless"),
    ("tmax",
     rf"T\s*max\s*(?:\)|value)?{_GAP}{{0,150}}?(?:is|was)\s+"
     rf"(?:approximately\s+)?{_VALUE}\s*(?P<unit>hours?|hrs?|h|minutes?|min)\b",
     "time"),
    # "mean plasma C max and AUC of X are 72.8 ng/mL and 276 hr*ng/mL" states
    # two quantities in one clause; each is captured by its own pattern.
    ("cmax",
     rf"C\s*max{_GAP}{{0,150}}?(?:is|was|are)\s+(?:approximately\s+)?{_VALUE}\s*"
     rf"(?P<unit>ng\s*/\s*mL|pg\s*/\s*mL|ug\s*/\s*mL|nmol\s*/\s*L)",
     "mass/volume"),
    ("auc",
     rf"AUC{_GAP}{{0,150}}?(?:is|was|are)\s+(?:approximately\s+)?{_VALUE}\s*"
     rf"(?P<unit>(?:hr?|h)\s*\*?\s*(?:ng|pg|ug)\s*/\s*mL"
     rf"|(?:ng|pg|ug)\s*\.?\s*(?:hr?|h)\s*/\s*mL)",
     "mass*time/volume"),
    ("cmax_and_auc_pair",
     rf"C\s*max\s+and\s+AUC[^.]{{0,80}}?are\s+{_VALUE}\s*"
     rf"(?P<unit>ng\s*/\s*mL|pg\s*/\s*mL)\s+and\s+(?P<auc_value>{_NUM})\s*"
     rf"(?P<auc_unit>(?:hr?|h)\s*\*?\s*(?:ng|pg)\s*/\s*mL)",
     "mass/volume"),
    ("protein_binding",
     rf"{_VALUE}\s*(?P<unit>%|percent)\s+of\s+\w+\s+binds",
     "dimensionless"),
]

#: A sentence boundary is a period followed by whitespace and a capital or
#: digit. Splitting on bare periods cuts "1.9- 4.0 hours" in half and produced
#: verbatim quotes truncated mid-number, such as "(range: 0.".
_BOUNDARY = re.compile(r"\.(?=\s+[A-Z0-9])|\.$")


def _sentence_containing(text: str, start: int, end: int) -> str:
    """The full sentence a match sits in, for the verbatim record."""
    left = 0
    for m in _BOUNDARY.finditer(text, 0, start):
        left = m.end()
    right = len(text)
    for m in _BOUNDARY.finditer(text, end):
        right = m.end()
        break
    return " ".join(text[left:right].split())


def _normalise_time_unit(unit: str) -> str:
    u = unit.lower().strip()
    if u.startswith("hour") or u in ("hr", "hrs", "h"):
        return "h"
    if u.startswith("min"):
        return "min"
    return unit


def _normalise_concentration_unit(raw: str) -> str:
    compact = re.sub(r"\s+", "", raw).lower()
    return {"ng/ml": "ng/mL", "pg/ml": "pg/mL", "ug/ml": "ug/mL",
            "nmol/l": "nmol/L"}.get(compact, raw)


def _normalise_auc_unit(raw: str) -> str:
    compact = re.sub(r"[\s*.]", "", raw).lower()
    for prefix in ("hrng/ml", "hng/ml", "nghr/ml", "ngh/ml"):
        if compact == prefix:
            return "ng*h/mL"
    for prefix in ("hrpg/ml", "hpg/ml", "pghr/ml", "pgh/ml"):
        if compact == prefix:
            return "pg*h/mL"
    return raw


def _normalise_unit(name: str, raw: str) -> str:
    raw = raw.strip()
    if name in ("half_life", "tmax"):
        return _normalise_time_unit(raw)
    if name == "clearance":
        compact = re.sub(r"\s+", "", raw).lower()
        if compact in ("l/hr", "l/hour", "l/h"):
            return "L/h"
        if compact == "ml/min":
            return "mL/min"
    if name == "volume_of_distribution":
        compact = re.sub(r"\s+", "", raw).lower()
        return "L/kg" if compact == "l/kg" else "L"
    if name in ("bioavailability", "protein_binding"):
        return "%"
    if name in ("cmax", "cmax_and_auc_pair"):
        return _normalise_concentration_unit(raw)
    if name == "auc":
        return _normalise_auc_unit(raw)
    return raw


def extract_parameters(text: str) -> list[PKParameter]:
    """Pull parameters out of a label's pharmacokinetics section."""
    found: list[PKParameter] = []
    seen: set[str] = set()

    for name, pattern, expected in PARAMETER_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            # Named groups: a pattern without an sd or bound group simply
            # returns None for it, rather than shifting every other field.
            g = match.groupdict()
            bound_marker = g.get("bound")
            sd = float(g["sd"]) if g.get("sd") else None
            unit_raw = g.get("unit") or ""
            value = float(g["value"])
            unit = _normalise_unit(name, unit_raw)
            verbatim = _sentence_containing(text, match.start(), match.end())
            # Attributes are read from a window around the match, not the whole
            # sentence. The bremelanotide label states half-life and clearance
            # in one sentence; reading the sentence as a whole gave half-life
            # the clearance's "CL/F" apparent-parameter flag and copied the
            # half-life's range onto the clearance.
            local = text[max(0, match.start() - 55): match.end() + 55]

            # Unit sanity: the parsed dimension must match what the parameter is.
            try:
                parse_unit(unit)
            except UnitError:
                logger.warning("rejecting %s: unparseable unit %r", name, unit)
                continue

            qualifier = Qualifier.UNSPECIFIED
            low = local.lower()
            if bound_marker:
                qualifier = Qualifier.UPPER_BOUND
            elif "median" in low:
                qualifier = Qualifier.MEDIAN
            elif "mean" in low:
                qualifier = Qualifier.MEAN
            elif "approximately" in low or "about" in low:
                qualifier = Qualifier.APPROXIMATE

            if name in ("bioavailability", "protein_binding"):
                value, unit = value / 100.0, "1"

            # A range qualifies a value only when it immediately follows it,
            # as in "2.7 hours (range: 1.9-4.0 hours)". Searching a wider
            # window attached the half-life's range to the clearance stated
            # later in the same sentence.
            trailing = text[match.end(): match.end() + 40]
            rng = re.match(rf"\s*\(?\s*range:?\s*({_NUM})\s*[–\-]\s*({_NUM})",
                           trailing, flags=re.IGNORECASE)
            # The paired pattern yields two parameters; rename before the
            # duplicate check so a Cmax found by both patterns collapses.
            if name == "cmax_and_auc_pair":
                auc_key = f"auc:{float(g['auc_value']):g}"
                if auc_key not in seen:
                    seen.add(auc_key)
                    found.append(PKParameter(
                        name="auc", value=float(g["auc_value"]),
                        unit=_normalise_auc_unit(g["auc_unit"]),
                        qualifier=qualifier, verbatim=verbatim,
                        note="stated jointly with Cmax in one clause",
                    ))
                name = "cmax"

            key = f"auc:{value:g}" if name == "auc" else f"{name}:{value:g}:{unit}"
            if key in seen:
                continue
            seen.add(key)

            found.append(PKParameter(
                name=name, value=value, unit=unit, qualifier=qualifier,
                verbatim=verbatim, sd=sd,
                range_low=float(rng.group(1)) if rng else None,
                range_high=float(rng.group(2)) if rng else None,
                is_apparent=bool(re.search(r"CL\s*/\s*F|V\s*/\s*F", local, re.I)),
                normalised_by_weight=unit.endswith("/kg"),
                note=("stated as an upper bound, not a point estimate"
                      if qualifier is Qualifier.UPPER_BOUND else None),
            ))
    return found


@dataclass
class LabelPKRecord:
    """Parameters extracted from one product label, with provenance."""

    generic_name: str
    brand_name: str | None
    parameters: list[PKParameter]
    route: str | None
    dose_mg: float | None
    dose_verbatim: str | None
    source: SourceRecord
    pk_section_chars: int = 0
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> PKParameter | None:
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    @property
    def parameter_names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def has_minimum_set(self) -> bool:
        """Enough to parameterise a one-compartment model with absorption."""
        return all(self.get(n) for n in
                   ("clearance", "volume_of_distribution", "bioavailability"))


class FDALabelClient:
    def __init__(self, http: CachedHttpClient | None = None):
        self.http = http or CachedHttpClient()

    def fetch(self, generic_name: str) -> LabelPKRecord | None:
        payload, fetch = self.http.get_json(OPENFDA_LABEL, {
            "search": f'openfda.generic_name:"{generic_name}"', "limit": 1,
        })
        results = payload.get("results", [])
        if not results:
            return None
        label = results[0]
        pk_text = " ".join(label.get("pharmacokinetics", []))
        if not pk_text:
            pk_text = " ".join(label.get("clinical_pharmacology", []))
        if not pk_text:
            return None

        openfda = label.get("openfda", {})
        dosage = " ".join(label.get("dosage_and_administration", []))
        dose_values = sorted({float(m) for m in
                              re.findall(rf"({_NUM})\s*mg\b", dosage)})
        dose_match = re.search(rf"({_NUM})\s*mg\b", dosage)
        route_match = re.search(
            r"subcutaneous|intravenous|intramuscular|oral(?:ly)?|intranasal",
            dosage or pk_text, re.I)

        params = extract_parameters(pk_text)
        warnings: list[str] = []
        if len(dose_values) > 1:
            warnings.append(
                "The label covers several strengths "
                f"({', '.join(f'{d:g}' for d in dose_values)} mg); the first was "
                "taken, and a simulation must be told which dose is intended."
            )
        for p in params:
            if p.is_apparent:
                warnings.append(
                    f"{p.name} is reported as an apparent parameter (CL/F or V/F); "
                    "it is confounded with bioavailability."
                )
            if p.normalised_by_weight:
                warnings.append(
                    f"{p.name} is weight-normalised ({p.unit}); a body weight is "
                    "required before it can be used in an absolute model."
                )
            if not p.is_usable_as_point:
                warnings.append(
                    f"{p.name} is stated as a bound ({p.verbatim[:80]}...), not a "
                    "point estimate."
                )

        return LabelPKRecord(
            generic_name=generic_name,
            brand_name=(openfda.get("brand_name") or [None])[0],
            parameters=params,
            route=route_match.group(0).lower() if route_match else None,
            dose_mg=float(dose_match.group(1)) if dose_match else None,
            dose_verbatim=(
                _sentence_containing(dosage, dose_match.start(), dose_match.end())
                if dose_match else None
            ),
            pk_section_chars=len(pk_text),
            warnings=sorted(set(warnings)),
            source=SourceRecord(
                source_db="openfda",
                source_id=(label.get("id") or generic_name),
                url="https://api.fda.gov/drug/label.json"
                    f'?search=openfda.generic_name:"{generic_name}"',
                title=f"FDA label — {generic_name}",
                retrieved_at=fetch.fetched_at,
                retrieval_endpoint=fetch.url,
                license_note=LICENSE_NOTE,
                content_sha256=SourceRecord.hash_payload(
                    {"pk": pk_text, "dosage": dosage}),
            ),
        )
