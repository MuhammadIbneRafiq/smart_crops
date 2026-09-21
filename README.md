# smart_crops — FieldCrop-SWE benchmark code

Notebooks and evaluation code accompanying the FieldCrop-SWE paper: a benchmark of modern
segmentation models on field-condition crop imagery collected in Sweden.

---

## Dataset — FieldCrop-SWE v1.0

620 RGB images collected in Swedish field conditions:

| crop | images | source resolutions |
|---|---|---|
| cabbage | 241 | 186 × 720·1280, 55 × 3840·2160 |
| purple kale | 155 | 111 × 3840·2160, 44 × 1920·1080 |
| yellow onion | 224 | 114 × 3840·2160, 110 × 720·1280 |
| **total** | **620** | |

Annotations are per-instance polygons in YOLO segmentation format
(`class x1 y1 x2 y2 … xn yn`, normalised), with frozen `train` / `valid` / `test` splits per crop.

The image pool spans substantial variation in spatial resolution, viewpoint, orientation and
image quality — which is why input handling is specified explicitly below rather than left to
each framework's default.

Data: [`muhammadibnerafiq/smart-crops`](https://www.kaggle.com/datasets/muhammadibnerafiq/smart-crops) on Kaggle.

FieldCrop-SWE v1.0 is the immutable dataset release used for the experiments in the paper.
Future expansions receive new version identifiers so the benchmark reported here stays
reproducible.

---

## Models benchmarked

| model | backbone | what is trained | task |
|---|---|---|---|
| YOLO26s-seg | — | full network | instance |
| Mask2Former | R50 | full network, from COCO instance weights | instance |
| SAM 2.1 | Hiera-S, frozen | prompt encoder + mask decoder | promptable |
| DINOv2 | ViT-B/14, frozen | linear head (`BNHead`, ~3.1k params) | semantic |
| ViT-MAE + adapters | ViT-B/16, frozen | bottleneck adapters + seg head (~2M params) | semantic |

---

## Evaluation protocol

Two metric families are reported. **They are not interchangeable, and models from different
families must not be placed in the same column.**

**Instance segmentation** — COCO mask AP (`AP50:95`, `AP50`, `AP75`) via `pycocotools`,
equivalent to what `yolo segment val` reports as `mAP50-95(M)` / `mAP50(M)`. Applies to
YOLO26s-seg, Mask2Former, and detector→SAM 2 pipelines.

**Semantic segmentation** — foreground IoU and Dice (crop vs background), plus pixel accuracy.
IoU and Dice are reported **aggregate** (intersection and union pooled across the whole split,
the standard semantic-seg figure) and as a per-image mean. Applies to DINOv2 and ViT-MAE.

To place an instance model on the semantic axis, union its per-image predicted instance masks
into one binary mask and score it identically. Going the other way — converting a semantic mask
to instances with connected components — merges touching plants and produces misleading AP, so
it is not used here.

### SAM 2 is reported separately

SAM 2 is promptable and has no class head, so it cannot localise objects on its own. It is
evaluated in **GT-Bbox mode** (ground-truth boxes supplied as prompts), which is a recognised
protocol but confers an oracle advantage: recall is perfect by construction. Those figures
are a mask-quality ceiling and are reported in their own block.

For an end-to-end number comparable to the other models, a detector is chained in front
(YOLO boxes → SAM 2 masks), which is the standard composition.

---

## Input resolution

All models use a **maximum input size of 1280×1280 with the original aspect ratio preserved**
by letterboxing or padding. Images are never stretched to a square.

This matters because the dataset mixes 16:9 landscape (3840×2160, 1920×1080) with 9:16 portrait
(720×1280) inside the same crop class — a square resize distorts those two groups in opposite
directions. 1280 retains considerably more detail than 640 for small and densely packed
instances while keeping compute manageable.

Per framework:

- **YOLO26s-seg** — `imgsz=1280`. Ultralytics letterboxes by default; `imgsz` sets the long side.
- **Mask2Former** — detectron2's large-scale-jitter mapper, aspect-preserving by construction.
- **DINOv2 / ViT-MAE** — mmsegmentation-style pipeline: scale-jitter the short side by a ratio
  range of (0.5, 2.0) with aspect preserved, then a fixed-size random crop. At test time the
  short side is resized with aspect preserved, so images vary in size and evaluation runs at
  batch 1. For ViT-B/14 the side must be a multiple of 14 (e.g. 1288 = 92 × 14).
- **SAM 2** — resizes internally to 1024×1024 by design; not configurable.

---

## Repository contents

```
notebooks/    Kaggle notebooks, one per model family
README.md
```

Each notebook is self-contained: it installs its dependencies, locates the dataset under
`/kaggle/input`, converts the YOLO polygon labels to the format the model needs, trains per
crop, and writes a results CSV to `/kaggle/working`.

---

## Reproducing

1. Create a Kaggle notebook and import the `.ipynb` you want.
2. Settings: **GPU T4 ×2** or **P100**, **Internet ON**.
3. Add the `smart-crops` dataset as an input.
4. Run top to bottom.

The notebooks locate the dataset by globbing `/kaggle/input` for `*/train/images` and matching
each crop by keyword, so they are insensitive to the exact mount path.

Label conversion handles both polygon lines (`class x1 y1 … xn yn`) and 4-number box lines
(`class cx cy w h`), and keeps images with no annotations as valid background examples.

---

## Citation

```bibtex
@article{fieldcropswe,
  title   = {FieldCrop-SWE},
  author  = {Rafiq, Muhammad Ibne},
  year    = {2026}
}
```

*Update this block with the final title, author list and venue before submission.*

---

## License

Code in this repository is released under the MIT License. The FieldCrop-SWE dataset is
distributed under its own terms — see the dataset card on Kaggle.
