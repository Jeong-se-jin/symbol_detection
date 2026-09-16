AP1000 v48 + Exact-Axis Near-Witness Short-Side Fallback
=========================================================

Purpose
-------
This version starts from the ORIGINAL v48 implementation. The v48 core is not
changed. One narrow fallback runs only after v48 has completed.

Fallback requirements
---------------------
1. Two raw H/V boundary runs must be on the EXACT same raster axis.
2. They must be adjacent runs on that exact axis; no same-axis run is skipped.
3. One side must satisfy the original v48 pipe-witness threshold.
4. The short side must be >= 70% of the original witness but still below it.
   On the validation sheet: original witness = 31 px, short minimum = 22 px.
5. Gap must be component-sized: 1.0 to 2.5 original witness lengths.
   On the validation sheet: 31 to 78 px.
6. Both endpoints must have exact off-axis attachment evidence.
7. The original v45 CONNECTED bridge_component must prove that one local
   foreground component bridges both endpoints.
8. A fallback candidate already represented by original v48 is discarded.

Validation on AP1000 Sheet 1
----------------------------
Original v48 accepted: 140
Modified accepted:     142
Original 140 changed:  0
New accepted:          2

New #140: V033A
  axis: 2273 -> 2273
  pipe support: 63 px + 23 px
  displacement: 4461 -> 4519
  bbox: (2247, 4461) - (2288, 4519)
  geometry: CONNECTED

New #141: V033B
  axis: 4023 -> 4023
  pipe support: 77 px + 29 px
  displacement: 4475 -> 4533
  bbox: (3996, 4475) - (4038, 4533)
  geometry: CONNECTED

V041 remains unrecognized by this fallback because its two raw carrier sides are
not on the exact same raster axis. No accepted bbox was added around V041.

Run
---
1. Run INSTALL_REQUIREMENTS.bat once.
2. Drag/drop a P&ID image or PDF onto RUN_V48_EXACT_AXIS_SHORT_SIDE.bat.
3. Results are written to RESULT_V48_EXACT_AXIS_SHORT_SIDE.
