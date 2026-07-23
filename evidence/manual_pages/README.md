# ROMO-B manual evidence pages

These JPEGs were rendered from the user-provided 368.6 MB ROMO-B PDF at 2400-pixel page scale
and JPEG quality 88. The complete 38-page compressed manual is stored at
[`source_files/hardware/ROMO-B_manual_complete_compressed.pdf`](../../source_files/hardware/ROMO-B_manual_complete_compressed.pdf).

| Page | File | Evidence |
|---:|---|---|
| 3 | [ROMO-B_page-03.jpg](ROMO-B_page-03.jpg) | ±30° steering limit, L/W dimensions, 2WIS/4WIS/Pivot examples |
| 26 | [ROMO-B_page-26.jpg](ROMO-B_page-26.jpg) | Ackermann derivation; uses 0.390 m steering-knuckle-center distance instead of 0.529 m track |
| 31 | [ROMO-B_page-31.jpg](ROMO-B_page-31.jpg) | manual states ideal ±45° but fixes Pivot at ±30°; wheel speed signs |
| 38 | [ROMO-B_page-38.jpg](ROMO-B_page-38.jpg) | 50 ms cycle, scalar packet fields, wheel feedback, stated errors |

The source scan stores page 38 sideways. Its evidence JPEG is rotated 90 degrees clockwise for
readability; the source content is otherwise unchanged.

The manual's ±45° statement is not adopted without checking its own geometry. Across the complete
PDF, 323 mm (or 0.323 m) appears six times on pages 2, 5, 26, and 35, while 324 mm (or 0.324 m)
appears once on page 3. The curated documents therefore use \(L=0.323\) m as the default and retain
0.324 m only as an isolated source outlier. With \(W_k=0.390\) m, the centered zero-slip angle is
approximately ±39.63°.
