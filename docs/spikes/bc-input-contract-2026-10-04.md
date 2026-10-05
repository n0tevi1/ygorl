# BC input integrity repair (2026-10-04)

Related: #88, #83. The legacy b2s/128×2 data was unbound to an environment;
all ten old demonstration decks violate the current MD banlist/card-pool rules.
The old reports also contained different processed sample counts. These findings
motivate repairing the input chain before capacity experiments, not relabeling
old artifacts or claiming their historical results measured modern MD strength.

## Repaired behavior

- `train_bc.py --seed` now seeds network construction as well as optimization.
  Previously it seeded only after constructing the new network.
- The BC command checks the complete solver environment stamp, including the
  fingerprint, and validates each demonstrated deck under that environment.
  `solve_openings.py` validates the input decks before scheduling solver jobs.
- `greedy_demos.py --env` records with that environment's rules and legal decks.
  Encoded artifacts identify their environment, ordered card-vocabulary hash,
  event window and format. Both `--extra` and `--extra-heldout` validate identity
  before use. Previously the command discarded their metadata and checked only
  array compatibility. Legacy unmarked arrays remain readable for audit through
  `load_data`, but must be regenerated for use by the training command.
- A native script-budget error can return a `DuelResult(reason="error")` without
  a Python exception. The heuristic recorder now drops that game's entire prefix,
  reports the error and makes the recording command exit nonzero. Previously
  those samples could enter training as if recording had succeeded.

## Validation and limits

BC/solver tests: **44 passed**. Regressions check identical trained weights for
the same CLI seed despite different caller RNG states, different weights for a
different seed, rejection of swapped card indices at equal vocabulary size,
environment revision/window mismatches, unmarked training and held-out arrays,
relabeled illegal decks, and engine-error prefixes.

A small end-to-end fixture uses legal MD `blue-eyes` and `branded` decks. Twelve
opening hands produce nine verified lines; an independent replay checks all
**389 steps with zero failures**. Split by hand index (0–2 train, 3–5 held-out),
not by selecting successful hands. Four training and four separate-seed held-out
Greedy games supply battle examples. Mixed training uses 135 solver + 111 battle
samples; held-out has 57 solver + 295 battle samples. A 32×1, one-epoch CPU run
produces an environment-bound checkpoint. This validates the pipeline only:
it is not a capacity comparison, a competitive teacher, or the complete #88 data.

The opening solver still uses its documented passive synthetic opponent and
known-deck-order setup. Those are single-player opening exercises, not legal
two-player MD matches or evidence of success under random shuffles/interruption.
The heuristic games use two legal decks and ordinary shuffled play. Larger BC
work must specify teacher coverage and freeze identical processed data across
network sizes before spending the long-training budget.

Artifacts: `out/research/bc-input-contract-2026-10-04/`. Final implementation
`3e2eb82`: full CPU presubmit **1,409 passed / 8 skipped** (323.28 s); separate
real ROCm tests **6 passed** (5.38 s), covering all five GPU-skipped cases.
The three remaining skips require optional solver/cache assets. There are no
hosted GitHub checks. `validation.json` pins the source and fixture artifacts;
the final `validated/` rerun reproduces all normal heuristic samples and every
mixed-BC parameter exactly, with eight zero-error heuristic games. This checks
that rejecting error prefixes leaves successful recording unchanged.
