"""Seed registry: the research collection as *naming facts only*.

This file deliberately contains no pharmacology. Each entry carries a canonical
key, the name to search, search aliases, a research area for organisation, and
a modality *hypothesis*. Chemistry — weight, formula, structure, development
phase — is resolved from ChEMBL and PubChem at runtime and attributed to those
sources, so nothing here can leak an unsourced scientific claim into the graph.

``declared_modality`` is a hypothesis, not an assertion: the resolver records
what the databases report and the two are compared. ``identity_note`` is
restricted to identity and nomenclature caveats, which are statements about
naming rather than about biology.

Entries flagged ``is_mixture`` are formulations, not molecular entities, and are
short-circuited before any structure lookup. Entries flagged
``supplier_defined`` have identities set by vendors rather than by a reference
database and require an analytical certificate before any classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.registry.resolve import Modality

__all__ = ["SeedEntry", "SEED_REGISTRY", "by_key", "by_area", "RESEARCH_AREAS"]


@dataclass(frozen=True, slots=True)
class SeedEntry:
    key: str
    name: str
    research_area: str
    declared_modality: Modality = Modality.UNKNOWN
    aliases: tuple[str, ...] = ()
    is_mixture: bool = False
    supplier_defined: bool = False
    reported_components: tuple[str, ...] = ()
    identity_note: str | None = None

    def __post_init__(self) -> None:
        if self.reported_components and not self.is_mixture:
            raise ValueError(f"{self.key}: components listed on a non-mixture entry")


RESEARCH_AREAS = (
    "metabolic_and_energy_regulation",
    "tissue_repair_and_immune_signaling",
    "dermal_and_copper_associated",
    "endocrine_and_growth_signaling",
    "neurobiology_and_cognition",
    "other_molecular_research",
)

_METABOLIC = "metabolic_and_energy_regulation"
_REPAIR = "tissue_repair_and_immune_signaling"
_DERMAL = "dermal_and_copper_associated"
_ENDOCRINE = "endocrine_and_growth_signaling"
_NEURO = "neurobiology_and_cognition"
_OTHER = "other_molecular_research"


SEED_REGISTRY: tuple[SeedEntry, ...] = (
    # -- metabolic and energy regulation --------------------------------------
    SeedEntry(
        "mots-c", "MOTS-c", _METABOLIC, Modality.PEPTIDE,
        aliases=("MOTS-c", "MOTSc", "mitochondrial open reading frame of the 12S rRNA-c"),
        identity_note="A mitochondrial-derived peptide; distinct from nuclear-encoded peptides.",
    ),
    SeedEntry(
        "5-amino-1mq", "5-Amino-1MQ", _METABOLIC, Modality.SMALL_MOLECULE,
        aliases=("5-amino-1-methylquinolinium", "5-amino-1-methylquinolinium iodide"),
        identity_note=(
            "A quinolinium small molecule, not a peptide. Name searches return "
            "unrelated aminocarboxylic acids, so resolution requires the full "
            "chemical name."
        ),
    ),
    SeedEntry("cagrilintide", "cagrilintide", _METABOLIC, Modality.PEPTIDE_HORMONE_ANALOG,
              aliases=("AM833", "NNC0174-0833")),
    SeedEntry("aod-9604", "AOD-9604", _METABOLIC, Modality.MODIFIED_PEPTIDE,
              aliases=("AOD9604", "AOD 9604"),
              identity_note="Described as a fragment analogue of human growth hormone."),
    SeedEntry("retatrutide", "retatrutide", _METABOLIC, Modality.MODIFIED_PEPTIDE,
              aliases=("LY3437943",)),
    SeedEntry("semaglutide", "semaglutide", _METABOLIC, Modality.PEPTIDE_HORMONE_ANALOG,
              aliases=("NN9535", "NNC0113-0217")),
    SeedEntry("tirzepatide", "tirzepatide", _METABOLIC, Modality.PEPTIDE_HORMONE_ANALOG,
              aliases=("LY3298176",)),
    SeedEntry("apelin", "apelin", _METABOLIC, Modality.PEPTIDE,
              aliases=("APLN", "apelin-13", "apelin-17", "apelin-36"),
              identity_note=(
                  "Apelin exists as multiple cleavage products of differing length; "
                  "a bare 'apelin' label does not identify which isoform was studied."
              )),

    # -- tissue repair and immune signaling -----------------------------------
    SeedEntry("bpc-157", "BPC-157", _REPAIR, Modality.PEPTIDE,
              aliases=("BPC 157", "pentadecapeptide BPC 157", "PL 14736", "Bepecin")),
    SeedEntry("tb-500", "TB-500", _REPAIR, Modality.PEPTIDE, supplier_defined=True,
              aliases=("TB500", "TB 500"),
              identity_note=(
                  "Marketed TB-500 is a vendor-defined fragment and is not "
                  "interchangeable with full-length thymosin beta-4; the exact "
                  "sequence must be read from the product's certificate of analysis."
              )),
    SeedEntry("thymosin-beta-4", "thymosin beta-4", _REPAIR, Modality.PEPTIDE,
              aliases=("TMSB4X", "thymosin β4", "Tβ4"),
              identity_note="Full-length endogenous peptide; held separately from TB-500."),
    SeedEntry("bpc157-tb500-blend", "BPC-157 / TB-500 blend", _REPAIR, Modality.MIXTURE,
              is_mixture=True, reported_components=("bpc-157", "tb-500")),
    SeedEntry("kpv", "KPV", _REPAIR, Modality.PEPTIDE,
              aliases=("lysine-proline-valine", "Lys-Pro-Val"),
              identity_note="Reported as a C-terminal tripeptide fragment of α-MSH."),
    SeedEntry("glow", "GLOW blend", _REPAIR, Modality.MIXTURE, is_mixture=True,
              reported_components=("ghk-cu", "bpc-157", "tb-500"),
              identity_note=(
                  "Components are as commonly reported by vendors; ratios are not "
                  "standardised and vary between products."
              )),
    SeedEntry("klow", "KLOW blend", _REPAIR, Modality.MIXTURE, is_mixture=True,
              reported_components=("ghk-cu", "bpc-157", "tb-500", "kpv"),
              identity_note=(
                  "Components are as commonly reported by vendors; ratios are not "
                  "standardised and vary between products."
              )),
    SeedEntry("ll-37", "LL-37", _REPAIR, Modality.PEPTIDE,
              aliases=("cathelicidin LL-37", "CAP-18", "hCAP18", "CAMP")),
    SeedEntry("cartalax", "cartalax", _REPAIR, Modality.PEPTIDE, aliases=("AED peptide",)),
    SeedEntry("ara-290", "ARA-290", _REPAIR, Modality.PEPTIDE,
              aliases=("cibinetide", "ARA 290"),
              identity_note="Described as a non-haematopoietic erythropoietin-derived peptide."),
    SeedEntry("thymosin-alpha-1", "thymosin alpha-1", _REPAIR, Modality.PEPTIDE,
              aliases=("thymalfasin", "thymosin α1", "Ta1", "Zadaxin")),
    SeedEntry("amino-h2o", "Amino H2O", _REPAIR, Modality.UNKNOWN, supplier_defined=True,
              identity_note=(
                  "A product name, not a chemical identity. No pharmacological "
                  "classification may be assigned until composition is verified; "
                  "it may be a diluent rather than an active substance."
              )),

    # -- dermal and copper-associated -----------------------------------------
    SeedEntry("ghk-cu", "GHK-Cu", _DERMAL, Modality.METAL_COMPLEX,
              aliases=("copper tripeptide-1", "GHK copper", "glycyl-histidyl-lysine copper"),
              identity_note=(
                  "A copper coordination complex. Copper speciation and the "
                  "peptide-to-copper ratio are part of its identity, and evidence "
                  "from topical use does not transfer to systemic exposure."
              )),
    SeedEntry("ahk-cu", "AHK-Cu", _DERMAL, Modality.METAL_COMPLEX,
              aliases=("alanyl-histidyl-lysine copper", "copper tripeptide AHK"),
              identity_note="Distinct tripeptide from GHK-Cu; evidence is not shared."),
    SeedEntry("snap-8", "SNAP-8", _DERMAL, Modality.MODIFIED_PEPTIDE,
              aliases=("acetyl octapeptide-3", "acetyl glutamyl heptapeptide-3"),
              identity_note="A cosmetic topical ingredient; systemic exposure is uncharacterised."),
    SeedEntry("collagen-peptides", "collagen-derived peptides", _DERMAL, Modality.MIXTURE,
              is_mixture=True,
              aliases=("hydrolyzed collagen", "collagen hydrolysate", "collagen tripeptide"),
              identity_note=(
                  "A heterogeneous hydrolysate category, not a defined molecule; "
                  "molecular weight distribution differs by product."
              )),

    # -- endocrine and growth signaling ---------------------------------------
    SeedEntry("tesamorelin", "tesamorelin", _ENDOCRINE, Modality.PEPTIDE_HORMONE_ANALOG,
              aliases=("TH9507", "Egrifta")),
    SeedEntry("sermorelin", "sermorelin", _ENDOCRINE, Modality.PEPTIDE_HORMONE_ANALOG,
              aliases=("GRF 1-29", "GHRH 1-29", "Geref")),
    SeedEntry("ipamorelin", "ipamorelin", _ENDOCRINE, Modality.PEPTIDE,
              aliases=("NNC 26-0161",)),
    SeedEntry("cjc-1295-dac", "CJC-1295 with DAC", _ENDOCRINE, Modality.MODIFIED_PEPTIDE,
              aliases=("CJC-1295 DAC", "DAC:GRF", "CJC1295 with DAC"),
              identity_note=(
                  "Carries a drug-affinity-complex modification intended to extend "
                  "circulation; a chemically different entity from the non-DAC form "
                  "and the two must not share evidence."
              )),
    SeedEntry("cjc-1295-no-dac", "CJC-1295 without DAC", _ENDOCRINE, Modality.MODIFIED_PEPTIDE,
              aliases=("modified GRF 1-29", "mod GRF 1-29", "CJC-1295 no DAC"),
              identity_note=(
                  "Vendor usage of the name 'CJC-1295' for this non-DAC analogue is "
                  "inconsistent; identity must be read per product."
              )),
    SeedEntry("cjc1295nodac-ipamorelin", "CJC-1295 without DAC + ipamorelin", _ENDOCRINE,
              Modality.MIXTURE, is_mixture=True,
              reported_components=("cjc-1295-no-dac", "ipamorelin")),
    SeedEntry("cjc1295dac-ipamorelin", "CJC-1295 with DAC + ipamorelin", _ENDOCRINE,
              Modality.MIXTURE, is_mixture=True,
              reported_components=("cjc-1295-dac", "ipamorelin"),
              identity_note="Listed as a distinct hypothetical research combination."),
    SeedEntry("kisspeptin", "kisspeptin", _ENDOCRINE, Modality.PEPTIDE,
              aliases=("KISS1", "metastin", "kisspeptin-10", "kisspeptin-54"),
              identity_note="Multiple fragment lengths are in use and are not equivalent."),
    SeedEntry("igf-1-lr3", "IGF-1 LR3", _ENDOCRINE, Modality.PROTEIN,
              aliases=("Long R3 IGF-1", "LR3 IGF-1", "IGF-I LR3"),
              identity_note="An analogue of IGF-1, not identical to endogenous IGF-1."),

    # -- neurobiology and cognition -------------------------------------------
    SeedEntry("semax", "Semax", _NEURO, Modality.MODIFIED_PEPTIDE,
              aliases=("ACTH 4-10 Pro-Gly-Pro", "Met-Glu-His-Phe-Pro-Gly-Pro")),
    SeedEntry("selank", "Selank", _NEURO, Modality.MODIFIED_PEPTIDE,
              aliases=("tuftsin Pro-Gly-Pro",)),
    SeedEntry("dihexa", "Dihexa", _NEURO, Modality.MODIFIED_PEPTIDE,
              aliases=("PNB-0408", "N-hexanoic-Tyr-Ile-(6)-aminohexanoic amide")),
    SeedEntry("pinealon", "Pinealon", _NEURO, Modality.PEPTIDE, aliases=("EDR peptide",)),
    SeedEntry("vip", "vasoactive intestinal peptide", _NEURO, Modality.PEPTIDE,
              aliases=("VIP", "aviptadil", "VIP peptide")),

    # -- other molecular research ---------------------------------------------
    SeedEntry("nad", "NAD+", _OTHER, Modality.NUCLEOTIDE_COFACTOR,
              aliases=("nicotinamide adenine dinucleotide", "NADIDE", "NAD"),
              identity_note=(
                  "A nucleotide cofactor, not a peptide. Oxidised (NAD+) and reduced "
                  "(NADH) forms, and the NAD+/NADH ratio, are distinct quantities; "
                  "extracellular administration does not imply equivalent change in "
                  "any intracellular compartment."
              )),
    SeedEntry("glutathione", "glutathione", _OTHER, Modality.UNKNOWN,
              aliases=("GSH", "L-glutathione", "gamma-glutamylcysteinylglycine"),
              identity_note=(
                  "A tripeptide conventionally handled as a small molecule; reduced "
                  "(GSH) and oxidised (GSSG) forms are distinct. Modality is left to "
                  "database resolution."
              )),
    SeedEntry("dsip", "DSIP", _OTHER, Modality.PEPTIDE,
              aliases=("delta sleep-inducing peptide", "delta-sleep inducing peptide")),
    SeedEntry("epitalon", "epitalon", _OTHER, Modality.PEPTIDE,
              aliases=("epithalon", "AEDG peptide", "Ala-Glu-Asp-Gly"),
              identity_note=(
                  "A synthetic tetrapeptide. It is NOT epithalamin, which is a pineal "
                  "extract; evidence for the two must never be pooled."
              )),
    SeedEntry("epithalamin", "epithalamin", _OTHER, Modality.MIXTURE, is_mixture=True,
              aliases=("epithalamine", "pineal peptide extract"),
              identity_note=(
                  "A bovine pineal gland extract of undefined composition, held as a "
                  "separate entry from the synthetic tetrapeptide epitalon."
              )),
    SeedEntry("pt-141", "PT-141", _OTHER, Modality.MODIFIED_PEPTIDE,
              aliases=("bremelanotide", "PT 141", "Vyleesi")),
    SeedEntry("melanotan-1", "melanotan I", _OTHER, Modality.MODIFIED_PEPTIDE,
              aliases=("melanotan-1", "MT-1"),
              identity_note=(
                  "Marketed 'melanotan I' must not be assumed identical to the "
                  "approved analogue afamelanotide; the exact molecular form "
                  "requires per-product verification, so afamelanotide is "
                  "deliberately not listed as an alias."
              )),
    SeedEntry("melanotan-2", "melanotan II", _OTHER, Modality.MODIFIED_PEPTIDE,
              aliases=("melanotan-2", "MT-II", "MT-2")),
)


def by_key(key: str) -> SeedEntry:
    for entry in SEED_REGISTRY:
        if entry.key == key:
            return entry
    raise KeyError(f"no seed registry entry with key {key!r}")


def by_area(area: str) -> tuple[SeedEntry, ...]:
    return tuple(e for e in SEED_REGISTRY if e.research_area == area)
