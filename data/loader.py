"""
data/loader.py
────────────────────────────────────────────────────
Load and filter the dataset for any crystal family.
Family configuration comes from crystal_families.py.

NOTE (publication): `api_key` must be supplied by the caller (e.g. via an
environment variable or a local, git-ignored config file). Never hardcode
a Materials Project API key in source that will be published.
"""
import json
import numpy as np
from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifWriter
from sklearn.preprocessing import LabelEncoder

from config import (XRD_NPY, METADATA_JSON, DATA_DIR,
                    SEQ_LEN, TWO_THETA_MIN, TWO_THETA_MAX,
                    ACTIVE_FAMILY)
from data.crystal_families import get_family


# ── Label extraction ─────────────────────────────────────────────

def get_label(formula: str, label_elements: list) -> str:
    """
    Return the label for a formula based on the family's label_elements.
    Finds the first element in the priority list that appears in the formula.
    """
    for el in label_elements:
        if el in formula:
            return el
    return "Other"


# ── Build dataset from Materials Project ─────────────────────────

def build_dataset(api_key: str):
    """
    Query Materials Project, simulate XRD patterns, and save the dataset.
    Intended to run once — subsequent calls load from the cached file.
    """
    family = get_family(ACTIVE_FAMILY)
    calc   = XRDCalculator(wavelength="CuKa")
    grid   = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, SEQ_LEN)

    print(f"Building dataset: {family['name']} ({family['formula']})")
    print(f"Space group: {family['space_group']}")

    with MPRester(api_key) as mpr:
        results = mpr.materials.search(
            spacegroup_number=family["space_group"],
            fields=["material_id", "structure", "formula_pretty"],
            **family["query"],
        )

    print(f"Found {len(results)} structures from Materials Project")

    patterns, metadata = [], []

    for doc in results:
        try:
            struct  = doc.structure
            formula = doc.formula_pretty.replace("/", "-").replace(" ", "")
            mat_id  = doc.material_id

            # Simulate the XRD pattern
            pat = calc.get_pattern(struct)
            vec = np.zeros(SEQ_LEN)
            for pos, inten in zip(pat.x, pat.y):
                if TWO_THETA_MIN <= pos <= TWO_THETA_MAX:
                    idx = np.argmin(np.abs(grid - pos))
                    vec[idx] = max(vec[idx], inten)

            patterns.append(vec)
            metadata.append({
                "material_id": mat_id,
                "formula":     formula,
                "is_virtual":  False,
                "origin":      formula,
                "replaced_from": None,
                "replaced_to":   None,
            })

        except Exception as e:
            print(f"  Skip {doc.formula_pretty}: {e}")

    # ── TER augmentation ────────────────────────────────────────
    replacement_map = family["replacement_map"]
    label_elements  = family["label_elements"]
    seen_formulas   = {m["formula"] for m in metadata}
    n_real          = len(metadata)

    for i in range(n_real):
        struct = results[i].structure
        elems  = [str(el) for el in struct.composition.elements
                  if str(el) != "O"]

        for old_el in elems:
            if old_el not in replacement_map:
                continue
            for new_el in replacement_map[old_el]:
                try:
                    new_struct = struct.copy()
                    new_struct.replace_species({old_el: new_el})
                    new_formula = new_struct.composition.reduced_formula
                    new_formula = new_formula.replace("/", "-").replace(" ", "")

                    if new_formula in seen_formulas:
                        continue

                    # Simulate the XRD pattern for the virtual structure
                    pat = calc.get_pattern(new_struct)
                    vec = np.zeros(SEQ_LEN)
                    for pos, inten in zip(pat.x, pat.y):
                        if TWO_THETA_MIN <= pos <= TWO_THETA_MAX:
                            idx = np.argmin(np.abs(grid - pos))
                            vec[idx] = max(vec[idx], inten)

                    patterns.append(vec)
                    metadata.append({
                        "material_id":   f"virtual_{len(metadata)}",
                        "formula":       new_formula,
                        "is_virtual":    True,
                        "origin":        metadata[i]["formula"],
                        "replaced_from": old_el,
                        "replaced_to":   new_el,
                    })
                    seen_formulas.add(new_formula)

                except Exception:
                    continue

    n_virtual = len(metadata) - n_real
    print(f"TER: {n_real} real + {n_virtual} virtual = {len(metadata)} total")

    # ── Save ────────────────────────────────────────────────────
    np.save(XRD_NPY, np.array(patterns))
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {DATA_DIR}/")
    return np.array(patterns), metadata


# ── Load dataset ─────────────────────────────────────────────────

def load_dataset(api_key: str = None) -> dict:
    """
    Load the dataset from disk if cached, otherwise build it from
    Materials Project. Returns a dict consumed by the rest of the pipeline.
    """
    import os

    family         = get_family(ACTIVE_FAMILY)
    label_elements = family["label_elements"]
    min_real       = family["min_real_samples"]

    # Build if not already cached
    if not os.path.exists(XRD_NPY):
        if api_key is None:
            raise ValueError(
                f"Dataset not found at {XRD_NPY}.\n"
                f"An API key is required to build it from Materials Project."
            )
        build_dataset(api_key)

    # Load
    patterns = np.load(XRD_NPY)
    with open(METADATA_JSON) as f:
        metadata = json.load(f)

    labels_raw = [get_label(m["formula"], label_elements)
                  for m in metadata]
    is_virtual = [m["is_virtual"] for m in metadata]

    # Filter: keep only classes with enough real samples
    real_count = {}
    for l, v in zip(labels_raw, is_virtual):
        if not v:
            real_count[l] = real_count.get(l, 0) + 1

    valid_classes = {
        c for c, n in real_count.items()
        if n >= min_real and c != "Other"
    }

    valid_idx  = [i for i, l in enumerate(labels_raw)
                  if l in valid_classes]
    patterns   = patterns[valid_idx]
    labels_raw = [labels_raw[i] for i in valid_idx]
    is_virtual = [is_virtual[i] for i in valid_idx]

    le      = LabelEncoder()
    labels  = le.fit_transform(labels_raw)
    classes = le.classes_

    print(f"\n[{family['name']}] Dataset: {len(labels)} samples | "
          f"{len(classes)} classes: {list(classes)}")

    # Print per-class stats
    for cls in classes:
        idxs   = [i for i, l in enumerate(labels_raw) if l == cls]
        n_real = sum(1 for i in idxs if not is_virtual[i])
        n_virt = len(idxs) - n_real
        print(f"  {cls:4s}: {n_real:3d} real + {n_virt:3d} virtual")

    origins_raw = [m.get("origin", m["formula"]) for m in metadata]
    origins_raw = [origins_raw[i] for i in valid_idx]  # apply the same filter

    return {
        "patterns":    patterns,
        "labels":      labels,
        "is_virtual":  np.array(is_virtual),
        "origins":     origins_raw,
        "classes":     classes,
        "le":          le,
        "n_classes":   len(classes),
        "family_name": ACTIVE_FAMILY,
        "family_cfg":  family,
    }
