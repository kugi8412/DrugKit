# `smiles_processing` — Technical Reference

DrugKit's RDKit-free preprocessing backend.  
Converts raw SMILES strings into `torch_geometric.data.Data` objects ready for the `GINEConv`-based `SiameseRankNet` model.

---

## Why this module exists

The rest of DrugKit (training, virtual screening, inference) historically called RDKit directly inside ad-hoc `smiles_to_graph_gine()` functions scattered across scripts. This module replaces that dependency with a pure-Python pipeline that is:

- **Deterministic** — identical SMILES always produces identical tensors.
- **Fault-tolerant** — a `strict=False` mode skips bad tokens instead of crashing, which matters at billion-scale (ZINC, ChEMBL, PubChem).
- **Dimensionally compatible** — atom features are 42-d and bond features are 11-d, matching the exact layout the existing GINEConv models were trained with.

---

## Module structure

```
smiles_processing/
├── smiles_errors.py       # Exception hierarchy
├── smiles_tokenizer.py    # SMILES string → list of tokens
├── smiles_parser.py       # Token list → molecular graph dict
├── smiles_features.py     # Graph dict → enriched graph dict (hybridization etc.)
├── feature_encoding.py    # Enriched graph → fixed-size float vectors
└── smiles_to_pyg.py       # End-to-end entry point → torch_geometric.data.Data
```

Each file is a pure transformation with no shared mutable state. The files form a strict linear dependency chain — each one only imports from files earlier in the list.

---

## The pipeline, step by step

```
SMILES string
     │
     ▼
smiles_tokenizer.tokenize_smiles()
     │   Splits the string into typed tokens:
     │   bare atoms  "C" "Cl" "c"
     │   bond chars  "=" "#" "/" "\"
     │   brackets    {"token_type":"bracket_atom", "symbol":"N", ...}
     │   ring digits "1" … "9"
     │   branch      "(" ")"
     ▼
smiles_parser.parse_smiles()
     │   Walks the token list with a branch stack + ring-closure dict.
     │   Produces a graph dict:
     │     {"atoms": [...], "bonds": [...]}
     │   Runs three post-processing passes:
     │     1. Promote implicit SINGLE bonds to AROMATIC where both atoms are aromatic
     │     2. Mark in_ring=True via DFS cycle detection
     │     3. Mark conjugated=True for bonds adjacent to any π bond
     ▼
smiles_features.extract_features()
     │   Iterates every atom and bond and computes:
     │     atom: hybridization (SP / SP2 / SP3) from bond environment
     │     bond: passes through in_ring, conjugated, stereochemistry
     │   Modifies the graph in-place (sets atom["hybridization"]).
     ▼
feature_encoding.encode_atom() / encode_bond()
     │   Maps each enriched atom/bond dict to a fixed-length float list.
     │   Atom → 42 floats   Bond → 11 floats
     │   Uses the same one-hot vocabularies as the original RDKit functions.
     ▼
smiles_to_pyg.smiles_to_pyg()
     │   Stacks atom vectors into x (N×42).
     │   Builds bidirectional COO edge_index and edge_attr (2E×11).
     │   Wraps everything in torch_geometric.data.Data with
     │   x, edge_index, edge_attr, smiles, y, is_elite.
     ▼
torch_geometric.data.Data  ← GINEConv model input
```

---

## File-by-file reference

### `smiles_errors.py`

Defines the exception hierarchy. All exceptions descend from `SMILESError` so callers can catch the whole family with one `except` clause.

| Exception | Raised when |
|---|---|
| `SMILESValidationError` | Input string is empty, or the parse produces zero atoms |
| `SMILESTokenizationError` | An unrecognised character or malformed bracket atom is encountered |
| `UnsupportedSMILESFeatureError` | A valid-but-unsupported SMILES feature is detected (disconnected `.`, reaction `>`, wildcard `*`) |
| `SMILESParseError` | Token stream is structurally invalid (unmatched parens, unclosed rings) |

---

### `smiles_tokenizer.py`

**Input:** raw SMILES string  
**Output:** `list[str | dict]`

Scans left-to-right with a hand-written positional loop (no regex on the main path). The key design decision is that bracket atoms `[...]` are handled entirely here and emitted as a single dict token; the parser never sees the raw bracket syntax.

**Bracket atom parsing** is done by `parse_bracket_atom()`, which applies one compiled regex to extract:

```
[<isotope>?  <symbol>  <chirality>?  <hcount>?  <charge>?  <map>?]
    └─ int     str        @/@@         H/H2       +/-n      ignored
```

The `strict` parameter changes error behaviour throughout:

- `strict=True` (default) — raises on the first problem. Use for development and validation.
- `strict=False` — skips unrecognised characters with a `logging.WARNING`. Use for large dataset preprocessing where a small fraction of malformed SMILES is acceptable.

---

### `smiles_parser.py`

**Input:** SMILES string (delegates tokenisation internally)  
**Output:** `{"atoms": [...], "bonds": [...]}`

Implements a recursive-descent-style token consumer using an explicit branch stack (iterative, not recursive). The main state machine tracks:

- `current_atom_idx` — the atom to bond the next atom to
- `branch_stack` — pushed on `(`, popped on `)` to restore the branching atom
- `ring_openings` — dict mapping ring-digit → (atom_idx, bond_token) for deferred ring-closure bonds
- `pending_bond` — the bond type to use for the next bond (defaults to SINGLE)

**Atom dict schema:**

```python
{
    "id":            int,        # zero-based index
    "symbol":        str,        # uppercase, e.g. "C", "FE"
    "formal_charge": int,        # 0 for most atoms; populated from brackets
    "hybridization": None,       # filled by smiles_features
    "chirality":     str | None, # "@", "@@", or None
    "aromatic":      bool,       # True if originated from lowercase or bracket aromatic
    "explicit_h":    int,        # from bracket atoms; 0 for bare atoms
}
```

**Bond dict schema:**

```python
{
    "start":          int,        # atom index
    "end":            int,        # atom index
    "bond_type":      str,        # "SINGLE" | "DOUBLE" | "TRIPLE" | "AROMATIC"
    "stereochemistry": str | None, # "/" | "\\" | None
    "in_ring":        bool,       # set by _mark_ring_bonds()
    "conjugated":     bool,       # set by _mark_conjugated_bonds()
}
```

---

### `smiles_features.py`

**Input:** raw graph dict from `parse_smiles`  
**Output:** enriched graph dict (same structure, `hybridization` fields filled)

Adds heuristic hybridisation to each atom by inspecting its incident bonds:

| Rule | Result |
|---|---|
| Atom is aromatic | SP2 |
| Has a triple bond, or two or more double bonds | SP |
| Has exactly one double bond | SP2 |
| All single bonds | SP3 |

This is sufficient for the common organic/drug-like subset. Chemically exact hybridisation (e.g. for metals, hypervalent sulfur) is outside scope.

`extract_features(graph)` is a convenience wrapper that processes the whole graph in one call and is the normal entry point. `extract_atom_features` and `extract_bond_features` operate on single atoms/bonds and are exposed for testing.

---

### `feature_encoding.py`

**Input:** enriched atom/bond dicts + full graph (needed to count degree)  
**Output:** `list[float]` of fixed length

This is the compatibility layer with the existing GNN code. The vocabulary lists and one-hot dimensions exactly mirror the original `get_atom_features` / `get_bond_features` functions in `improved_train.py` and `virtual_screeing.py`.

**Atom vector layout (42 dimensions):**

| Slice | Feature | Vocabulary size |
|---|---|---|
| `[0:13]` | Atom symbol one-hot | 12 permitted + 1 unknown |
| `[13:19]` | Graph degree one-hot | 0–4 + unknown |
| `[19:25]` | Total H count one-hot | 0–4 + unknown |
| `[25:31]` | Formal charge one-hot | −1, −2, +1, +2, 0 + unknown |
| `[31:37]` | Hybridisation one-hot | SP, SP2, SP3, SP3D, SP3D2 + unknown |
| `[37]` | Is aromatic | 0.0 / 1.0 |
| `[38]` | Atomic mass × 0.01 | float |
| `[39:42]` | Chirality one-hot | CW, CCW + unknown/none |

**Bond vector layout (11 dimensions):**

| Index | Feature |
|---|---|
| `[0]` | Is SINGLE |
| `[1]` | Is DOUBLE |
| `[2]` | Is TRIPLE |
| `[3]` | Is AROMATIC |
| `[4]` | Is conjugated |
| `[5]` | Is in ring |
| `[6:11]` | Stereo one-hot (STEREOZ, STEREOE, STEREOCIS, STEREOTRANS + none) |

Atoms not in `PERMITTED_ATOMS` (e.g. Fe, Cu, Zn) activate the unknown bucket at index 12 for the symbol slot. All other features still encode correctly, so metal-centred bracket atoms produce valid — if chemically approximate — feature vectors.

---

### `smiles_to_pyg.py`

**Input:** SMILES string + optional labels  
**Output:** `torch_geometric.data.Data` or `None`

This is the single entry point that all DrugKit pipelines should use. It chains all five steps above and handles exceptions.

```python
# Single molecule
data = smiles_to_pyg("CCO", y=-7.5, is_elite=True, strict=False)
# data.x          → FloatTensor (N, 42)
# data.edge_index → LongTensor  (2, 2E)
# data.edge_attr  → FloatTensor (2E, 11)
# data.smiles     → "CCO"
# data.y          → tensor([-7.5])
# data.is_elite   → tensor([1.0])

# Batch — tolerant, returns only successes
data_list, failed = batch_smiles_to_pyg(
    smiles_list,
    labels=scores,
    elite_set={"CCO"},
    strict=False,
)
```

Edges are stored bidirectionally (each bond appears as both `(start→end)` and `(end→start)`) to satisfy the undirected message-passing convention used by `GINEConv`.

**Migrating from the old RDKit functions:**

```python
# Before (in improved_train.py / virtual_screeing.py):
from rdkit import Chem
def smiles_to_graph_gine(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ...

# After — drop-in replacement, identical output dimensions:
from smiles_processing import smiles_to_pyg
def smiles_to_graph_gine(smiles, selectivity=None, is_elite=False):
    return smiles_to_pyg(smiles, y=selectivity, is_elite=is_elite, strict=False)
```

---

## Test coverage

Tests live in `tests/` and are organised to mirror the module structure.

| Test file | What it covers |
|---|---|
| `test_tokenizer.py` | Bare atoms, bracket atoms, `parse_bracket_atom`, tolerant mode, strict error cases |
| `test_parser.py` | Graph topology, aromaticity, ring closures, bracket atoms end-to-end, conjugation, stereo bonds, tolerant parsing |
| `test_feature_encoding.py` | Dimension constants, per-slot one-hot correctness, mass normalisation, chirality encoding, determinism |
| `test_smiles_to_pyg.py` | Schema validation, optional labels, GINEConv dimension contract, real-world drugs, tolerant/batch modes, edge cases |
| `test_regression.py` | Locked atom/bond counts for 15 reference molecules, encoding stability spot-checks, ZINC-sample corpus, idempotency |

Run with:

```bash
PYTHONPATH=. pytest tests/ -v
```

245 tests, all passing.

---

## What is explicitly not implemented

This is an ML preprocessing pipeline, not a chemistry toolkit. The following are out of scope by design:

- Full SMILES specification (extended ring closures `%nn`, disconnected `.`, reactions `>`, wildcards `*`)
- Exact valence chemistry and hydrogen count calculation
- SMARTS pattern matching
- Canonicalisation
- 3D coordinate generation
- Chemically exact stereochemistry resolution
- Isotope semantics (isotope numbers are parsed and stored but ignored downstream)