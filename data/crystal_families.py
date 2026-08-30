"""
crystal_families.py
────────────────────────────────────────────────────
Definitions of all supported crystal families.

To add a new family: copy one block and fill in the fields.
No other file needs to be modified.
"""

CRYSTAL_FAMILIES = {

    # ══════════════════════════════════════════════
    # SPINEL — AB₂O₄  |  SG 227 (Fd-3m)
    # ══════════════════════════════════════════════
    "spinel": {
        "name":          "Spinel",
        "formula":       "AB₂O₄",
        "space_group":   227,
        "description":   "Battery, magnetics, ceramic materials",

        # Materials Project query parameters
        "query": {
            "elements":     ["O"],
            "num_elements": (3, 3),
            "num_sites":    (14, 14),
        },

        # Which site to classify (B-site element)
        "label_elements": [
            "W", "Mo", "Sb", "Bi", "Al", "Cr", "Ga",
            "In", "Ti", "Sn", "V",  "Fe", "Mn",
        ],

        "min_real_samples": 3,

        # Ionic radius of label elements (Angstrom, octahedral coordination)
        "ionic_radius": {
            "Al": 0.535, "Bi": 1.030, "Cr": 0.615,
            "Fe": 0.645, "Ga": 0.620, "In": 0.800,
            "Mn": 0.645, "Mo": 0.590, "Sb": 0.600,
            "Sn": 0.690, "Ti": 0.605, "V":  0.640,
            "W":  0.600,
        },

        # TER replacement rules (same oxidation state, similar ionic radius)
        "replacement_map": {
            "Ca": ["Mg", "Sr", "Ba", "Zn"],
            "Mg": ["Ca", "Zn", "Fe", "Co", "Ni"],
            "Zn": ["Mg", "Ca", "Fe", "Co"],
            "Fe": ["Mg", "Zn", "Co", "Ni", "Mn"],
            "Co": ["Mg", "Zn", "Fe", "Ni"],
            "Ni": ["Mg", "Zn", "Fe", "Co"],
            "W":  ["Mo"],
            "Mo": ["W",  "Cr"],
            "Sb": ["Bi", "Nb", "Ta"],
            "Bi": ["Sb"],
            "Al": ["Cr", "Ga", "In"],
            "Cr": ["Al", "Fe", "V"],
            "Ga": ["Al", "In"],
            "Ti": ["Zr", "Hf", "Sn"],
            "Zr": ["Ti", "Hf"],
        },
    },

    "scheelite": {
        "name": "Scheelite",
        "formula": "ABO₄",
        "space_group": 88,
        "description": "Scintillators, laser media (CaWO4, PbWO4)",

        "query": {
            "elements": ["O"],
            "num_elements": (3, 3),
        },

        "label_elements": [
            "W", "Mo", "V", "Cr", "As",
            "Nb", "Ta", "P", "Mn", "Re",
        ],

        "min_real_samples": 2,
        "n_folds": 5,

        "ionic_radius": {
            "W": 0.600, "Mo": 0.590, "V": 0.540,
            "Cr": 0.440, "As": 0.475, "Nb": 0.640,
            "Ta": 0.640, "P": 0.380, "Mn": 0.530,
            "Re": 0.530,
        },

        "replacement_map": {
            # A-site (large 2+ cations, 8-fold coordination)
            "Ca": ["Sr", "Ba", "Pb", "Cd"],
            "Sr": ["Ca", "Ba", "Pb"],
            "Ba": ["Sr", "Ca", "Pb"],
            "Pb": ["Ca", "Sr", "Ba"],
            "Bi": ["La", "Nd"],
            "La": ["Nd", "Sm", "Bi"],
            "Nd": ["La", "Sm"],
            # B-site (tetrahedral, high oxidation state)
            "W": ["Mo", "Cr"],
            "Mo": ["W", "Cr"],
            "V": ["Nb", "As", "P"],
            "Nb": ["Ta", "V"],
            "Ta": ["Nb"],
            "As": ["V", "P"],
            "P": ["As", "V"],
        },
    },

    # ══════════════════════════════════════════════
    # GARNET — A₃B₂C₃O₁₂  |  SG 230 (Ia-3d)
    # ══════════════════════════════════════════════
    "garnet": {
        "name":          "Garnet",
        "formula":       "A₃B₂C₃O₁₂",
        "space_group":   230,
        "description":   "Laser media, solid-state electrolytes",

        "query": {
            "elements":     ["O"],
            "num_elements": (3, 4),
            "num_sites":    (40, 160),   # flexible: Z=2 -> 40, Z=8 -> 160
        },

        # Classify by B-site (octahedral, 3+ cation)
        "label_elements": [
            "Al", "Fe", "Ga", "Cr", "Sc", "In",
            "V",  "Mn", "Co", "Rh", "Ir",
        ],

        "min_real_samples": 2,   # smaller dataset for garnet

        # Ionic radius (octahedral, 3+ coordination)
        "ionic_radius": {
            "Al": 0.535, "Fe": 0.645, "Ga": 0.620,
            "Cr": 0.615, "Sc": 0.745, "In": 0.800,
            "V":  0.640, "Mn": 0.645, "Co": 0.545,
            "Rh": 0.665, "Ir": 0.625,
        },

        # TER: replace both A-site (large rare-earth) and B-site (octahedral)
        "replacement_map": {
            # A-site (dodecahedral, 3+): rare earth / alkaline earth
            "Y":  ["Gd", "Dy", "Ho", "Er", "Lu", "Sm"],
            "Gd": ["Y",  "Dy", "Nd", "Sm", "Tb"],
            "Nd": ["Sm", "Pr", "Gd", "La"],
            "La": ["Nd", "Pr"],
            "Lu": ["Yb", "Y"],
            "Yb": ["Lu", "Y",  "Er"],
            "Dy": ["Y",  "Ho", "Gd"],
            "Ca": ["Mg"],    # 2+ A-site garnets
            # B-site (octahedral, 3+)
            "Al": ["Cr", "Fe", "Ga", "Sc"],
            "Fe": ["Al", "Cr", "Ga"],
            "Ga": ["Al", "In"],
            "Sc": ["Al", "In", "Lu"],
            "Cr": ["Al", "Fe", "V"],
        },
    },
}


def get_family(family_name: str) -> dict:
    """Return the config dict for a given crystal family."""
    if family_name not in CRYSTAL_FAMILIES:
        available = list(CRYSTAL_FAMILIES.keys())
        raise ValueError(
            f"Family '{family_name}' not found.\n"
            f"Available: {available}"
        )
    return CRYSTAL_FAMILIES[family_name]


def list_families() -> list:
    """List all available families."""
    return [
        f"{k}: {v['formula']} (SG {v['space_group']}) — {v['description']}"
        for k, v in CRYSTAL_FAMILIES.items()
    ]
