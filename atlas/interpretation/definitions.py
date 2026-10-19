"""Definitions of the quantities ATLAS reports.

Each entry states what a quantity is, the mathematics behind it, what it does
*not* establish, and which measurements would test a claim built on it. The
"does not establish" field is the one that earns its place: most
misinterpretation in pharmacology comes from reading a real number as answering
a question it cannot answer -- affinity as efficacy, exposure as benefit,
plasma concentration as tissue concentration.

Nothing here is compound-specific. The engine in :mod:`atlas.interpretation.engine`
combines these definitions with an actual value and the metadata of the model
that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["EndpointDefinition", "DEFINITIONS", "get_definition"]


@dataclass(frozen=True)
class EndpointDefinition:
    key: str
    symbol: str
    name: str
    unit: str
    definition: str
    transformation: str
    establishes: list[str] = field(default_factory=list)
    does_not_establish: list[str] = field(default_factory=list)
    common_misreadings: list[str] = field(default_factory=list)
    validating_measurements: list[str] = field(default_factory=list)
    uncertainty_drivers: list[str] = field(default_factory=list)


DEFINITIONS: dict[str, EndpointDefinition] = {
    "pki": EndpointDefinition(
        key="pki", symbol="pKi", name="Negative log binding affinity", unit="log10 M",
        definition=(
            "The negative base-10 logarithm of the equilibrium inhibition "
            "constant Ki expressed in molar units. Higher values mean tighter "
            "binding."
        ),
        transformation="pKi = −log10(Ki [M]), so Ki [M] = 10^(−pKi)",
        establishes=[
            "How tightly the molecule binds the target under the assay's "
            "equilibrium conditions.",
            "A rank ordering against other ligands measured in the same assay.",
        ],
        does_not_establish=[
            "That binding produces any functional response. A tight-binding "
            "antagonist and a tight-binding agonist have the same pKi.",
            "That the molecule reaches the target in a living organism, which "
            "depends on exposure, permeability and stability.",
            "Selectivity. Affinity at one target says nothing about the others.",
            "Clinical effectiveness, which depends on exposure, duration, "
            "receptor reserve and the physiology downstream.",
        ],
        common_misreadings=[
            "Treating a one-unit pKi difference as a small difference. One unit "
            "is a tenfold change in Ki.",
            "Comparing pKi values measured in different assay systems or "
            "against different radioligands as if they were interchangeable.",
        ],
        validating_measurements=[
            "A radioligand displacement assay against the same target.",
            "A functional assay (EC50 or IC50) to establish whether binding "
            "produces a response, and in which direction.",
            "A selectivity panel across related receptor subtypes.",
        ],
        uncertainty_drivers=[
            "Assay variability between laboratories, commonly several tenths of "
            "a log unit even for the same compound.",
            "For predictions: how far the molecule sits from the training set.",
        ],
    ),
    "ki": EndpointDefinition(
        key="ki", symbol="Ki", name="Equilibrium inhibition constant", unit="nM",
        definition=(
            "The concentration of the molecule at which half the target sites "
            "are occupied at equilibrium, after correcting for the competing "
            "radioligand. Lower values mean tighter binding."
        ),
        transformation="Ki [nM] = 10^(9 − pKi)",
        establishes=["Binding affinity in concentration units."],
        does_not_establish=[
            "A functional effect, a tissue concentration, or a dose.",
            "That the concentration is achievable in vivo.",
        ],
        common_misreadings=[
            "Comparing a Ki to a plasma concentration without accounting for "
            "protein binding, which leaves only the unbound fraction available.",
        ],
        validating_measurements=[
            "Direct measurement by saturation binding (Kd).",
            "Measured unbound plasma concentration at the dose of interest.",
        ],
        uncertainty_drivers=[
            "Propagated from pKi; a symmetric interval in pKi becomes "
            "asymmetric in Ki because the transformation is logarithmic.",
        ],
    ),
    "auc": EndpointDefinition(
        key="auc", symbol="AUC", name="Area under the concentration-time curve",
        unit="ng·h/mL",
        definition=(
            "The integral of concentration over time. It summarises total "
            "systemic exposure across a dosing interval or to infinity."
        ),
        transformation=(
            "AUC(0,∞) = ∫C(t)dt. For a linear model this equals F·Dose/CL, so "
            "AUC is inversely proportional to clearance."
        ),
        establishes=[
            "Cumulative systemic exposure under the model or study conditions.",
            "Clearance, when dose and bioavailability are known.",
        ],
        does_not_establish=[
            "Benefit or toxicity. Exposure is not an outcome.",
            "Concentration at the target tissue, which can differ from plasma "
            "by orders of magnitude.",
            "That a given exposure is safe or effective at any dose.",
        ],
        common_misreadings=[
            "Treating AUC extrapolated to infinity as observed when a large "
            "fraction is extrapolated beyond the last sample.",
            "Comparing AUC across routes without accounting for bioavailability.",
        ],
        validating_measurements=[
            "A pharmacokinetic study with sampling through at least three "
            "terminal half-lives.",
            "Measured tissue concentrations, if target-site exposure is the "
            "question.",
        ],
        uncertainty_drivers=[
            "Clearance uncertainty propagates directly and inversely.",
            "The fraction of AUC obtained by extrapolation rather than observation.",
        ],
    ),
    "cmax": EndpointDefinition(
        key="cmax", symbol="Cmax", name="Maximum concentration", unit="ng/mL",
        definition=(
            "The highest concentration reached. In a model it is the peak of "
            "the simulated curve; in a study it is the highest sampled value, "
            "which underestimates the true peak unless sampling is dense."
        ),
        transformation=(
            "For one-compartment first-order absorption, Cmax occurs at "
            "tmax = ln(ka/ke)/(ka − ke) and depends on the balance between "
            "absorption and elimination."
        ),
        establishes=[
            "Peak systemic exposure, which is the relevant quantity for "
            "concentration-dependent effects.",
        ],
        does_not_establish=[
            "Target-tissue peak concentration.",
            "Peak effect, which may lag the peak concentration.",
        ],
        common_misreadings=[
            "Reading an observed Cmax as the true peak when sampling is sparse "
            "near tmax.",
            "Assuming a one-compartment model reproduces the peak; a sharp "
            "distribution phase produces a higher, earlier peak than such a "
            "model can generate.",
        ],
        validating_measurements=[
            "Dense early sampling around the expected tmax.",
            "A model comparison against a two-compartment structure.",
        ],
        uncertainty_drivers=[
            "The absorption rate constant, which is often derived from a "
            "reported tmax rather than measured directly.",
            "Model structure: peak height is the quantity most sensitive to it.",
        ],
    ),
    "tmax": EndpointDefinition(
        key="tmax", symbol="Tmax", name="Time of maximum concentration", unit="h",
        definition="The time at which the maximum concentration occurs.",
        transformation="tmax = ln(ka/ke)/(ka − ke); as ka → ke this tends to 1/ke.",
        establishes=["How quickly systemic exposure peaks after administration."],
        does_not_establish=[
            "Onset of effect, which depends on distribution to the target and "
            "on the downstream response.",
        ],
        common_misreadings=[
            "Treating a reported median tmax as a mean; its distribution is "
            "skewed and the median is the convention.",
        ],
        validating_measurements=["Dense sampling over the absorption phase."],
        uncertainty_drivers=[
            "Sampling schedule: an observed tmax can only fall on a sampled time.",
        ],
    ),
    "half_life": EndpointDefinition(
        key="half_life", symbol="t½", name="Terminal half-life", unit="h",
        definition=(
            "The time for concentration to fall by half during the terminal "
            "phase, once distribution is complete."
        ),
        transformation="t½ = ln(2)/λz, and for a one-compartment model λz = CL/V.",
        establishes=[
            "How long the compound persists in the sampled compartment.",
            "Time to steady state under repeated dosing: about 4–5 half-lives.",
        ],
        does_not_establish=[
            "Duration of biological effect. A compound can act long after it "
            "has cleared, through receptor kinetics or downstream signalling.",
            "Residence time at the target.",
        ],
        common_misreadings=[
            "Reading terminal half-life as the half-life of effect.",
            "In flip-flop kinetics, where absorption is slower than elimination, "
            "the terminal slope reflects absorption, not elimination.",
        ],
        validating_measurements=[
            "Sampling through at least three terminal half-lives.",
            "A pharmacodynamic time course, if duration of effect is the question.",
        ],
        uncertainty_drivers=[
            "Number and spacing of points in the terminal phase.",
            "Whether the apparent terminal phase is truly terminal.",
        ],
    ),
    "clearance": EndpointDefinition(
        key="clearance", symbol="CL", name="Clearance", unit="L/h",
        definition=(
            "The volume of plasma cleared of the compound per unit time. It "
            "determines steady-state exposure for a given dosing rate."
        ),
        transformation="CL = F·Dose/AUC; Css,avg = F·Dose/(CL·τ)",
        establishes=["Exposure for a given dosing rate, under linear kinetics."],
        does_not_establish=[
            "The route of elimination, or which organ performs it.",
            "Behaviour at doses where kinetics become non-linear.",
        ],
        common_misreadings=[
            "Treating an apparent clearance CL/F as a true clearance. It is "
            "confounded with bioavailability and is larger than CL whenever "
            "F < 1.",
        ],
        validating_measurements=[
            "An intravenous study, which gives CL without the F confound.",
            "Mass balance and metabolite identification.",
        ],
        uncertainty_drivers=["Between-subject variability; AUC estimation error."],
    ),
    "volume_of_distribution": EndpointDefinition(
        key="volume_of_distribution", symbol="V", name="Volume of distribution",
        unit="L",
        definition=(
            "The apparent volume required to contain the amount of drug in the "
            "body at the observed plasma concentration. It is a proportionality "
            "constant, not an anatomical volume."
        ),
        transformation="V = Amount/Concentration; t½ = ln2·V/CL",
        establishes=["The extent of distribution outside plasma."],
        does_not_establish=[
            "Where the compound goes. A large volume indicates tissue "
            "distribution without identifying the tissue.",
        ],
        common_misreadings=[
            "Reading a volume larger than total body water as impossible. It "
            "simply means extensive tissue binding.",
        ],
        validating_measurements=["Tissue distribution studies; imaging."],
        uncertainty_drivers=["Between-subject variability; assay sensitivity."],
    ),
    "bioavailability": EndpointDefinition(
        key="bioavailability", symbol="F", name="Bioavailability", unit="fraction",
        definition=(
            "The fraction of an administered dose reaching systemic circulation "
            "unchanged, relative to intravenous administration."
        ),
        transformation="F = (AUC_route/Dose_route)/(AUC_iv/Dose_iv)",
        establishes=["How much of the dose becomes systemically available."],
        does_not_establish=[
            "The rate of absorption, which is a separate property.",
            "Target-site availability.",
        ],
        common_misreadings=[
            "Using a reported upper bound such as 'less than 4%' as a point "
            "value.",
        ],
        validating_measurements=["A crossover study against intravenous dosing."],
        uncertainty_drivers=["Requires an intravenous reference arm to measure."],
    ),
    "mae": EndpointDefinition(
        key="mae", symbol="MAE", name="Mean absolute error", unit="log units",
        definition=(
            "The average absolute difference between predicted and observed "
            "values on held-out data."
        ),
        transformation=(
            "MAE = mean(|predicted − observed|). On a log scale an MAE of 1.0 "
            "means the typical prediction is off by tenfold in the underlying "
            "quantity."
        ),
        establishes=[
            "Typical prediction error on the evaluation set, in its units.",
        ],
        does_not_establish=[
            "Error on molecules unlike the evaluation set.",
            "That any individual prediction is within the MAE; it is an average.",
        ],
        common_misreadings=[
            "Comparing MAE across differently constructed splits. A random "
            "split gives a smaller MAE than a scaffold split on the same model, "
            "without the model being better.",
        ],
        validating_measurements=[
            "Evaluation on an independent external dataset.",
            "Prospective prediction followed by measurement.",
        ],
        uncertainty_drivers=[
            "Evaluation set size and how it was constructed.",
            "The experimental noise floor of the measurements themselves.",
        ],
    ),
    "coverage": EndpointDefinition(
        key="coverage", symbol="coverage", name="Empirical interval coverage",
        unit="fraction",
        definition=(
            "The fraction of held-out observations that fell inside their "
            "predicted interval."
        ),
        transformation=(
            "Coverage = mean(lower ≤ observed ≤ upper). It should match the "
            "nominal level; below it the intervals are too narrow."
        ),
        establishes=[
            "Whether intervals mean what they claim, on data like the "
            "evaluation set.",
        ],
        does_not_establish=[
            "Coverage under distribution shift. Conformal guarantees require "
            "calibration and test data to be exchangeable.",
            "Conditional coverage for any particular subgroup unless that "
            "subgroup was calibrated separately.",
        ],
        common_misreadings=[
            "Reading nominal coverage as achieved coverage.",
        ],
        validating_measurements=[
            "Coverage measured on an independent external set.",
        ],
        uncertainty_drivers=[
            "Calibration set size, and how similar it is to the data the "
            "intervals are applied to.",
        ],
    ),
    "tanimoto": EndpointDefinition(
        key="tanimoto", symbol="T", name="Tanimoto similarity", unit="fraction",
        definition=(
            "The size of the intersection of two molecules' fingerprint bits "
            "divided by the size of their union. 1.0 means identical "
            "fingerprints."
        ),
        transformation="T(A,B) = |A ∩ B| / |A ∪ B|",
        establishes=[
            "How much substructure a molecule shares with the training set, "
            "which is the basis of the applicability-domain check.",
        ],
        does_not_establish=[
            "Biological similarity. Two molecules can share most substructure "
            "and differ greatly in activity -- the activity cliff problem.",
            "That a similar molecule has a similar target.",
        ],
        common_misreadings=[
            "Treating high similarity as a guarantee of accurate prediction.",
        ],
        validating_measurements=[
            "Measured activity for the molecule in question.",
        ],
        uncertainty_drivers=[
            "Fingerprint choice and radius change the numbers substantially.",
        ],
    ),
    "r2": EndpointDefinition(
        key="r2", symbol="R²", name="Coefficient of determination", unit="fraction",
        definition=(
            "The fraction of variance in the held-out observations explained "
            "by the predictions, relative to predicting their mean."
        ),
        transformation="R² = 1 − SS_res/SS_tot",
        establishes=["How much better the model is than predicting the mean."],
        does_not_establish=[
            "Accuracy in absolute terms; a high R² on a wide-ranging set can "
            "still mean large errors.",
        ],
        common_misreadings=[
            "Treating a negative R² as impossible. It means the model is worse "
            "than predicting the test set's own mean.",
        ],
        validating_measurements=["Evaluation on an independent set."],
        uncertainty_drivers=[
            "The spread of the evaluation set: the same model scores a higher "
            "R² on a more variable set.",
        ],
    ),
    "receptor_occupancy": EndpointDefinition(
        key="receptor_occupancy", symbol="RO", name="Receptor occupancy",
        unit="fraction",
        definition=(
            "The fraction of target receptors bound at a given free "
            "concentration, under an equilibrium binding model."
        ),
        transformation="RO = C/(C + Kd), where C is the free concentration",
        establishes=[
            "The proportion of receptors engaged, under equilibrium assumptions.",
        ],
        does_not_establish=[
            "Functional response. Receptor reserve means a small occupancy can "
            "produce a maximal effect, and high occupancy can produce none if "
            "the ligand is an antagonist.",
            "That equilibrium has been reached, which fast kinetics may prevent.",
        ],
        common_misreadings=[
            "Using total rather than free concentration, which overstates "
            "occupancy by the protein-bound fraction.",
        ],
        validating_measurements=[
            "Receptor occupancy imaging, or a functional assay at matched "
            "concentrations.",
        ],
        uncertainty_drivers=[
            "Free concentration at the target, which is rarely measured.",
        ],
    ),
}


def get_definition(key: str) -> EndpointDefinition | None:
    return DEFINITIONS.get(key)
