# Provenance and limitations

## Evidence hierarchy

The bundle applies this evidence order:

1. User-provided hardware manuals and files
2. Independently recomputed geometry and equations
3. Peer-reviewed publisher/DOI records
4. Official journal scope pages
5. Oracle/Web ChatGPT as advisory review only

An AI review never overrides the manual, measured hardware data, or primary literature.

## Manual provenance

The original ROMO-B PDF supplied by the user is 368,621,615 bytes and exceeds GitHub's
100 MB single-file limit. The repository stores:

- a complete 38-page compressed derivative;
- high-resolution images of pages 3, 26, 31, and 38;
- the original filename, byte size, SHA-256, and local source path in the source manifest.

The source scan's page 38 is sideways. The compressed derivative and the page-38 evidence JPEG
rotate that page 90 degrees clockwise for readability without changing its content.

The compressed copy is for this private research repository. It is not represented as an
official vendor republication.

## Scholarly literature

No subscription-only article PDF is redistributed. The repository stores DOI metadata,
publisher links, BibTeX, and concise claim mappings. arXiv-only and MDPI papers are excluded
from the evidentiary basis in accordance with the researcher's instruction.

The search is substantial enough to identify direct novelty collisions, but it is not claimed
to be an exhaustive enumeration of every Google Scholar record.

## Oracle/Web ChatGPT record

`evidence/oracle/2026-07-21_slip_pivot_audit_raw.md` is the raw second-model advisory response.
It was requested as a Pro browser follow-up in a previously Pro-reviewed conversation. The
follow-up tool recorded `requested=Pro` but could not independently verify the picker badge
for that continuation (`resolved=(unavailable); verified=no`). Therefore the result is retained
for provenance but not described as independently verified server-side Pro output.

The curated documents correct or qualify the raw response where required and use the hardware
manual and peer-reviewed sources as ground truth.

## Unresolved hardware facts

- exact mass and 3D CoM of the approximately 55 kg front installation;
- actual steering-axis and wheel-contact geometry;
- steering compliance under load;
- wheel torque/current and tire-force access;
- synchronized dynamic corner loads;
- validated mounting and arm dynamic safety limits;
- high-bandwidth base/arm clock synchronization.

These are experimental prerequisites, not values to infer from the current documentation.
