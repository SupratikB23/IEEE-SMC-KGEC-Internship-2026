# Deep Learning for Medical Tissue Image Analysis
### Automatic Segmentation of Glands and Nuclei in Microscope Images

**IEEE Systems, Man, and Cybernetics Society — Student Branch Chapter, Kalyani Government Engineering College**
Summer Research Internship Programme 2026 · Computer Vision · 15 June – 15 August 2026

| | |
|---|---|
| **Student** | Supratik Bhowal — CSE (AIML), 3rd Year |
| **Mentor** | Prof. Animesh Hazra — Assistant Professor, CSE, Jalpaiguri Government Engineering College |

---

## What this project does

Pathologists grade cancer by counting and measuring individual structures — glands, nuclei — in
stained tissue viewed under a microscope. The work is slow, since one slide can hold thousands of
nuclei; it is subjective, since two qualified observers often disagree; and it does not scale, since
slides accumulate faster than trained pathologists do.

Automating it needs **instance segmentation**, which is harder than ordinary segmentation because
two problems must be solved at once: decide which pixels belong to tissue structures, *and* split
structures that physically touch into separate, countable objects.

Modern **pathology foundation models** — vision transformers pre-trained on millions of unlabelled
tissue images — are very good at the first problem and surprisingly bad at the second. Nothing in
their pre-training ever asks them to tell two touching objects apart, so they are effectively
boundary-blind: pixel accuracy stays high while every object-level count collapses. That gap is the
problem this project set out to close.

## How it was approached

The central decision was to **leave the pre-trained model completely frozen**. None of its original
weights are ever modified. Small trainable adapter layers are inserted instead, amounting to under
three percent of the parameters, and only those are trained. This costs far less memory, trains far
faster, and — importantly — cannot overwrite the knowledge that made the pre-trained model worth
using in the first place.

Those adapters are trained first on **unlabelled** tissue, using a signal read directly from the
staining itself. Hematoxylin binds to nuclei, so its intensity changes sharply wherever one cell
meets the next; the gradient of that channel marks object edges without a pathologist annotating
anything. Only afterwards is a compact decoder trained on annotated images to produce the final
outlines, followed by a post-processing step that separates structures the network merged.

Two constraints were treated as hard requirements throughout, not preferences:

- **A single free-tier cloud GPU** (NVIDIA T4, 16 GB), rather than the server-class hardware most of
  the literature assumes. Peak training memory stayed under 3 GB.
- **Few expert annotations**, because pathologist annotation is the dominant cost in this field.

Both were deliberate. A method needing neither expensive hardware nor large annotated datasets is
far likelier to be usable in a teaching hospital or a small research lab than one that needs both.

## What was found

Four results, described in full in [`docs/`](docs/):

**Adaptation improved object-level accuracy on all three datasets, while pixel accuracy barely
moved.** That gap is the informative part — it indicates the pre-trained model already knew *what*
the tissue was, and what adaptation supplied was the ability to tell *one object from another*.

**The gain was largest where objects were smallest and most densely packed** — precisely where
separating touching objects is hardest, which is consistent with the mechanism above.

**Unlabelled images cut the annotation requirement to roughly one fifth.** Since unlabelled tissue
already sits in every hospital archive while annotation is expensive, this is the finding with the
clearest practical consequence.

**Adaptation cost was a small fraction of comparable systems'** — a few megabytes stored per task
against the hundreds of megabytes a fully fine-tuned model requires, at comparable accuracy.

## Datasets

Three public benchmarks, chosen to differ in object size, density and dataset size, so that a method
working on all three is not tuned to one kind of object:

| Dataset | Structures | Why it was chosen |
|---|---|---|
| **GlaS** | Colorectal glands | Large objects, very small dataset |
| **PanNuke** | Cell nuclei | Many tissue types, large dataset |
| **CoNSeP** | Colorectal nuclei | Extremely densely packed objects |

The loaders in this release cover GlaS and PanNuke; the CoNSeP reader is part of the extended
study noted below.

All three are public, de-identified research releases. **No image data or trained weights are stored
in this repository** — PanNuke and CoNSeP carry non-commercial research licences that any derived
model inherits.

## Technologies used

Python 3.10+ with **PyTorch** for the model and training loops and **HuggingFace
`transformers`** to load the frozen Phikon-v2 backbone. **OpenCV**, **scikit-image** and **SciPy**
handle image processing, the watershed post-processing and the connected-component analysis behind
the instance metrics, with **NumPy** underneath. The stain normalisation, the adapters, every loss
and every metric are implemented directly rather than pulled from a library, which keeps the
dependency list short and the behaviour inspectable. Everything runs on a single NVIDIA T4.

## Repository layout

```
Medical-Tissue-Instance-Segmentation/
├── README.md
├── requirements.txt
├── docs/
│   ├── IEEE_SMC_KGEC_Final_Report.docx     Full internship report
│   ├── IEEE_SMC_KGEC_Presentation.pptx     Project presentation
│   └── RESULTS_SUMMARY.md                  Findings and experimental conditions
└── src/
    ├── config.py           All settings, overridable from the environment
    ├── stain.py            Stage 0 — Macenko normalisation, hematoxylin channel,
    │                       the annotation-free boundary target, tissue detection
    ├── data.py             Dataset loading, tiling, augmentation
    ├── model.py            Frozen backbone, DoRA adapters, decoders, predictor
    ├── losses.py           Self-supervised and supervised objectives
    ├── metrics.py          Instance extraction and evaluation (AJI, PQ, object Dice/F1)
    ├── phase0_prep.py      Stage 0 driver
    ├── phase1_ssl.py       Stage 1 — self-supervised adapter training
    ├── phase2_train.py     Stage 2 — supervised read-out and evaluation
    ├── semi_cotrain.py     Semi-supervised co-training
    └── run_all.py          Runs all three stages
```

The eleven modules are self-contained: `model.py` builds the adapters and decoder,
`losses.py` holds every objective, `metrics.py` does instance extraction and scoring, and the
four stage scripts are thin drivers over them. Nothing here depends on code outside `src/`.

> **Scope.** This is the pipeline as described in the report. The extended ablation studies,
> hyperparameter searches and the CoNSeP data loader used for the wider study are held back
> until the associated research paper is finalised.

## Setup

```bash
git clone https://github.com/<your-username>/Medical-Tissue-Instance-Segmentation.git
cd Medical-Tissue-Instance-Segmentation

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU with at least 16 GB. On Kaggle or Colab most dependencies are
pre-installed and only the remainder need adding.

The Phikon-v2 backbone is access-gated on HuggingFace: accept its terms once, then authenticate with
`huggingface-cli login`. **No token, key or password appears anywhere in this repository, and none
should ever be added** — `.gitignore` blocks the usual credential filenames as a safety net.

## Usage

Each stage runs on its own, or all of them in sequence. Behaviour is controlled entirely by
environment variables, so no source file needs editing between experiments.

```bash
cd src
python run_all.py --smoke     # a few images and epochs, verifies the wiring
python run_all.py             # the full pipeline

python phase0_prep.py         # Stage 0 — normalise, derive targets, cache
python phase1_ssl.py          # Stage 1 — train adapters on unlabelled tiles
python phase2_train.py        # Stage 2 — train the decoder and evaluate
python semi_cotrain.py --budget 16   # semi-supervised, 16 annotated images
```

Start with `--smoke`. It runs the whole pipeline on a handful of images for two epochs, so a
broken path or a missing dependency surfaces in a minute rather than an hour.

Common settings:

| Variable | Meaning | Default |
|---|---|---|
| `BJEPA_DATASETS` | Datasets to run, comma-separated | `glas` |
| `BJEPA_SEEDS` | Random seeds, comma-separated | `0` |
| `BJEPA_DECODER` | `unet` or `segformer` | `unet` |
| `BJEPA_FT_EPOCHS` | Supervised training epochs | `80` |
| `BJEPA_LORA_RANK` | Adapter rank | `8` |
| `BJEPA_MU` | Weight on the boundary term in Stage 1 | `0.3` |
| `BJEPA_GLAS_ROOT` | Path to the GlaS images | unset |
| `BJEPA_STATE_ROOT` | Where caches, adapters and results are written | `./bjepa_state` |

For example, three seeds on GlaS with the U-Net decoder:

```bash
BJEPA_DATASETS=glas BJEPA_SEEDS=0,1,2 BJEPA_DECODER=unet python phase2_train.py
```

Before any of this, point the code at your data:

```bash
export BJEPA_GLAS_ROOT=/path/to/glas        # Windows: $env:BJEPA_GLAS_ROOT = "D:\data\glas"
```

Each run writes a JSON record under `bjepa_state/stage2_results/`, named by dataset and seed and
carrying every metric alongside the configuration that produced it, so a number in the report can
always be traced back to the run behind it. `run_all.py` additionally writes a `summary.json` with
the mean and spread across seeds.

## A note on responsible use

This is a **research prototype for computational morphometry, not a diagnostic device.** It has no
regulatory clearance and must not be used to make or withhold a clinical decision. Its appropriate
role is assistive measurement reviewed by a qualified pathologist.

Instance errors are not neutral: an over-split inflates object counts and a merge deflates them, and
both propagate into the measurements used for grading. Any clinical pipeline built on this would
need calibration against expert counts on its own material.

## Acknowledgements

Carried out under the IEEE SMC Student Branch Chapter, KGEC — Research Internship Programme 2026,
mentored by Prof. Animesh Hazra. Built on the publicly released Phikon-v2 backbone and the GlaS,
PanNuke and CoNSeP datasets.

## References

1. K. Sirinukunwattana *et al.*, "Gland segmentation in colon histology images: The GlaS challenge contest," *Medical Image Analysis*, vol. 35, pp. 489–502, 2017.
2. J. Gamper *et al.*, "PanNuke: An open pan-cancer histology dataset for nuclei instance segmentation and classification," in *Proc. ECDP*, 2019, pp. 11–19.
3. S. Graham *et al.*, "HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images," *Medical Image Analysis*, vol. 58, art. 101563, 2019.
4. A. Filiot *et al.*, "Phikon-v2, a large and public feature extractor for biomarker prediction," arXiv:2409.09173, 2024.
5. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.
