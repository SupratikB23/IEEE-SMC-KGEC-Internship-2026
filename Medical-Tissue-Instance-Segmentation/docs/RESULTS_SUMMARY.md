# Results Summary

The findings below are those reported in the internship report. Full quantitative tables,
per-seed spreads and statistical tests belong to the associated research paper, which is still
being finalised, and are therefore not reproduced here.

---

## 1. Adaptation improved object-level accuracy on all three datasets

The adapted model separated objects far better than the unadapted pre-trained model, while
**pixel-level accuracy improved only slightly**.

That gap between the two is the informative part. It indicates that the pre-trained model could
already recognise *what* the tissue was — its pixel classification was close to correct all along —
and that what adaptation supplied was the ability to tell *one object from another*. Boundary
competence, not tissue recognition, was the thing missing.

This also explains why pixel-level metrics alone are a poor way to evaluate these models: a network
can score well on semantic Dice while merging every pair of touching objects, and every downstream
count would still be wrong.

## 2. The gain was largest where objects were small and densely packed

Improvement scaled with object density. The largest gain appeared on the densest dataset —
precisely the setting in which separating touching objects is hardest and in which a
boundary-blind model fails worst.

The ordering is consistent with the mechanism in Finding 1: the more crowded the objects, the more
of the achievable accuracy depends on boundaries rather than on tissue classification.

## 3. Unlabelled images substantially reduced the annotation requirement

Training on a small annotated set together with unlabelled images from the same dataset reached
the accuracy of the fully annotated setting using roughly **one fifth of the annotations**.

This is the finding with the clearest practical consequence. Annotation by a pathologist is the
dominant cost in this field, whereas unlabelled tissue images already exist in quantity in any
hospital archive. A method that converts the cheap resource into a substitute for the expensive
one changes what a small laboratory can realistically attempt.

The benefit was largest when annotations were scarcest and shrank as they accumulated, which is
what one would expect if the unlabelled data is genuinely supplying information the labels do not.

## 4. Adaptation cost was far lower than that of comparable systems

Existing state-of-the-art systems retrain tens or hundreds of millions of parameters for every new
task and store a correspondingly large model each time. The approach here retrained **under three
percent** of the model and stored a per-task module of a **few megabytes**, at comparable accuracy.

The distinction matters when a laboratory works across several organ types. Full fine-tuning stores
one large model per task; this approach stores one shared frozen backbone plus one small module per
task.

One qualification, stated plainly: the efficiency claim concerns **adaptation cost**, not raw
computation. The backbone is a large transformer and inference still costs what a large transformer
costs. What is small is the part that must be trained and stored per task.

---

## Experimental conditions

- All experiments ran on a **single free-tier NVIDIA T4 (16 GB)**, with peak training memory below
  **3 GB**.
- One fixed configuration was applied to all three datasets, with **no dataset-specific tuning**.
- Reported results are averaged over **multiple random seeds** (GlaS, CoNSeP) or **cross-validation
  folds** (PanNuke).
- Differences between configurations were assessed with **paired statistical tests**, not by
  comparing single numbers.

## Limitations acknowledged in the report

- The self-supervised stage helps mainly when annotations are scarce and when the unlabelled data
  matches the target domain; it is not an unconditional improvement.
- Results are reported class-agnostically. Distinguishing nucleus *types* is future work.
- The boundary signal is derived from hematoxylin staining and so assumes H&E-stained material.
- Evaluation operates on tiles and small native images rather than on whole slides.

## Planned future work

1. Multi-scale inputs, so that very large and very small structures are handled equally well.
2. Cell-type and nucleus-type classification alongside detection and segmentation.
3. Testing on other pre-trained backbones, to confirm the method is not specific to one.
4. Publication of the complete research paper.
