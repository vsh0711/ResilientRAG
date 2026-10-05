# Chunking strategy benchmark

Real PDFs, MiniLM dense + BM25 + cross-encoder rerank, top-3 retrieval. A hit means one of the three retrieved chunks contains the **entire** fact sentence, so a strategy that cuts a fact in two scores a miss. 40 questions per document, each asked in the document's wording and as a paraphrase (80 per cell below). Intervals are 95% Wilson.

The agent's own picks: **manual** → `structure`, **projects** → `recursive_character`, **fieldnotes** → `semantic`, **codebase** → `code`.

## Retrieval: `dense`

| document | fixed@500 | recursive_character@500 | semantic@500 | fixed@900 | recursive_character@900 | semantic@900 | recursive_character | semantic | auto |
|---|---|---|---|---|---|---|---|---|---|
| manual | 90% (81%–95%) | **100% (95%–100%)** | 99% (93%–100%) | 94% (86%–97%) | **100% (95%–100%)** | **100% (95%–100%)** | 90% (81%–95%) | 94% (86%–97%) | 100% (95%–100%) · picked `structure` |
| projects | 91% (83%–96%) | 95% (88%–98%) | 89% (80%–94%) | 85% (76%–91%) | **96% (90%–99%)** | 91% (83%–96%) | 88% (78%–93%) | 72% (62%–81%) | 88% (78%–93%) · picked `recursive_character` |
| fieldnotes | 79% (69%–86%) | **95% (88%–98%)** | 80% (70%–87%) | 72% (62%–81%) | 72% (62%–81%) | 89% (80%–94%) | 80% (70%–87%) | 85% (76%–91%) | 85% (76%–91%) · picked `semantic` |
| codebase | 94% (86%–97%) | **100% (95%–100%)** | 99% (93%–100%) | 86% (77%–92%) | 94% (86%–97%) | 82% (73%–89%) | 62% (52%–72%) | 72% (62%–81%) | 68% (57%–77%) · picked `code` |
| **all four** | 88% (84%–91%) | 98% (95%–99%) | 92% (88%–94%) | 84% (80%–88%) | 91% (87%–93%) | 91% (87%–93%) | 80% (75%–84%) | 81% (76%–85%) | 85% (81%–88%) |

## Retrieval: `hybrid_rerank`

| document | fixed@500 | recursive_character@500 | semantic@500 | fixed@900 | recursive_character@900 | semantic@900 | recursive_character | semantic | auto |
|---|---|---|---|---|---|---|---|---|---|
| manual | 95% (88%–98%) | **100% (95%–100%)** | **100% (95%–100%)** | 99% (93%–100%) | 99% (93%–100%) | **100% (95%–100%)** | **100% (95%–100%)** | **100% (95%–100%)** | 100% (95%–100%) · picked `structure` |
| projects | 98% (91%–99%) | **100% (95%–100%)** | 90% (81%–95%) | **100% (95%–100%)** | **100% (95%–100%)** | 98% (91%–99%) | 89% (80%–94%) | 88% (78%–93%) | 89% (80%–94%) · picked `recursive_character` |
| fieldnotes | 98% (91%–99%) | **100% (95%–100%)** | **100% (95%–100%)** | 95% (88%–98%) | **100% (95%–100%)** | **100% (95%–100%)** | 79% (69%–86%) | 99% (93%–100%) | 99% (93%–100%) · picked `semantic` |
| codebase | **100% (95%–100%)** | **100% (95%–100%)** | **100% (95%–100%)** | 98% (91%–99%) | **100% (95%–100%)** | 90% (81%–95%) | 89% (80%–94%) | 92% (85%–97%) | 84% (74%–90%) · picked `code` |
| **all four** | 98% (95%–99%) | 100% (99%–100%) | 98% (95%–99%) | 98% (96%–99%) | 100% (98%–100%) | 97% (94%–98%) | 89% (85%–92%) | 95% (92%–97%) | 93% (89%–95%) |

## Paraphrase only (the hard half), `hybrid_rerank`

| strategy | hits |
|---|---|
| fixed@500 | 98% (94%–99%) |
| recursive_character@500 | 100% (98%–100%) |
| semantic@500 | 98% (94%–99%) |
| fixed@900 | 99% (96%–100%) |
| recursive_character@900 | 100% (98%–100%) |
| semantic@900 | 97% (93%–99%) |
| recursive_character | 89% (83%–93%) |
| semantic | 95% (90%–97%) |
| auto | 94% (89%–97%) |

## Chunk counts

| document | fixed@500 | recursive_character@500 | semantic@500 | fixed@900 | recursive_character@900 | semantic@900 | recursive_character | semantic | auto |
|---|---|---|---|---|---|---|---|---|---|
| manual | 71 × 494 | 69 × 454 | 60 × 509 | 40 × 875 | 48 × 654 | 47 × 649 | 22 × 1418 | 39 × 783 | 48 × 637 |
| projects | 84 × 498 | 110 × 335 | 70 × 522 | 47 × 891 | 55 × 668 | 49 × 746 | 28 × 1312 | 40 × 915 | 28 × 1312 |
| fieldnotes | 73 × 494 | 75 × 420 | 62 × 505 | 41 × 881 | 39 × 814 | 44 × 713 | 23 × 1499 | 36 × 871 | 36 × 871 |
| codebase | 75 × 495 | 90 × 368 | 61 × 521 | 42 × 885 | 45 × 737 | 35 × 908 | 24 × 1384 | 23 × 1382 | 23 × 1418 |
