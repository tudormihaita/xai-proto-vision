# Prototype-Based Methods — Implementation & Output Reference

This document describes how each of the four prototype-based methods works,
what they produce during forward passes and training, how to visualise their
outputs, and what data from CUB-200 is required at each stage.

---

## Shared foundation

All four methods use the same backbone (ResNet-34 or VGG-16 pretrained on
ImageNet) and operate on its **spatial feature map**, not on a flattened vector.
For a 224×224 input the backbone produces a `(B, 512, 7, 7)` tensor — a 7×7
grid of 512-dimensional patch embeddings, where each cell corresponds to a
roughly 32×32 pixel receptive field in the original image. This spatial
structure is what makes prototype matching possible.

The backbone head (global average pool + FC classifier) is always removed.
Everything below plugs into the raw `(B, 512, 7, 7)` feature map.

---

## CUB-200 annotations — what each method actually uses

| Annotation file          | Used during training | Used for visualisation |
|---|---|---|
| `image_class_labels.txt` | All methods (supervision signal) | No |
| `bounding_boxes.txt`     | Optional pre-crop (all methods)  | No |
| `parts/part_locs.txt`    | **None**                         | Optional qualitative eval |
| `attributes/`            | **None**                         | No |

The spatial localisation you see in prototype explanations comes entirely from
the feature map — no part or bounding box annotations are used as targets.
Part annotations could optionally be used *after* training to measure whether
prototype regions align with human-labelled bird parts, but this is not
required by any of the four methods.

---

## ProtoPNet

**Paper:** Chen et al., NeurIPS 2019 — "This Looks Like That"

### Architecture

```
image → backbone → (B, 512, 7, 7) → PrototypeLayer → similarities → FC → logits
```

The prototype layer contains `P × num_classes` prototype vectors, each of
dimension 512. For each image it computes the minimum L2 distance between each
prototype and every patch in the 7×7 grid, then converts distances to
similarity scores via `log((min_dist + 1) / (min_dist + ε))`.

The FC classifier has a constrained weight matrix: class k can only receive
positive contributions from its own prototypes and negative contributions from
others.

### Training — three phases

**Phase 1 (warm-up, ~5 epochs):** train only the prototype layer and
classifier; backbone frozen.

**Phase 2 (joint, ~15 epochs):** train all components jointly with a combined
loss of three terms:
- `L_cls` — cross-entropy classification loss
- `L_cluster` — pushes each prototype close to at least one training patch
- `L_separation` — pushes prototypes of different classes apart

**Push step (once, after Phase 2):** scan the full training set, find the
actual training image patch with minimum L2 distance to each prototype, and
replace the prototype vector with that patch's embedding. This anchors
prototypes to real images.

**Phase 3 (fine-tune, ~10 epochs):** freeze backbone and prototypes; train
only the FC classifier.

### `training_step` loss dict

```python
{"total": loss, "cls": l_cls, "cluster": l_cluster, "separation": l_sep}
```

### `explain()` output

```python
{
    "prototype_similarities": Tensor,  # (P × num_classes,) — score per prototype
    "patch_locations":        Tensor,  # (P × num_classes, 2) — (row, col) in 7×7 grid
    "activation_maps":        Tensor,  # (P × num_classes, 7, 7) — raw activation per prototype
}
```

### Visualisation

For each of the top-K scoring prototypes for a given test image:

1. Take the `(7, 7)` activation map for that prototype.
2. Upsample to `(224, 224)` using bilinear interpolation.
3. Normalise to [0, 1] and apply a colormap (e.g. `plt.cm.hot`).
4. Overlay as a semi-transparent heatmap on the original image.
5. Draw a bounding box around the highest-activation region.
6. Alongside, display the training image patch the prototype was pushed to.

The reader sees pairs: *test image region ↔ training image region*.

### What you need from CUB

Only `image_class_labels.txt` and `train_test_split.txt`. Bounding boxes are
optional pre-processing. Nothing else.

---

## ProtoTree

**Paper:** Nauta et al., CVPR 2021 — "Neural Prototype Trees"

### Architecture

```
image → backbone → (B, 512, 7, 7) → global max pool → (B, 512)
      → tree of depth D with (2^D - 1) prototype nodes → leaf distributions → logits
```

Each internal tree node holds one prototype vector. At each node the routing
probability (probability of going RIGHT) is:

```
p_right = sigmoid(cosine_similarity(features, prototype))
p_left  = 1 - p_right
```

The path probability to any leaf is the product of routing probabilities along
the path. The class prediction is the expected leaf distribution weighted by
path probabilities.

Prototypes are shared across classes (unlike ProtoPNet), making the tree more
compact.

### Training

Joint training of backbone + all prototype nodes + leaf distributions.
Loss = cross-entropy over the expected leaf distribution.

**Push step (once, after training):** same as ProtoPNet — scan the training
set and anchor each node's prototype to the nearest real training patch. This
is required before deployment for interpretable visualisation.

### `training_step` loss dict

```python
{"total": loss, "cls": l_cls}
# optionally: {"total": ..., "cls": ..., "entropy": l_entropy}
# if a leaf-entropy regulariser is used
```

### `explain()` output

```python
{
    "routing_probs":    list[float],   # probability at each node along the greedy path
    "path_node_ids":    list[int],     # node indices visited (length = depth D)
    "leaf_reached":     int,           # index of the final leaf
    "node_similarities": Tensor,       # cosine similarity at each visited node
    "activation_maps":  list[Tensor],  # (7, 7) map per visited node
}
```

### Visualisation

Two complementary figures:

**Decision path figure:** a vertical or horizontal chain showing each visited
node. At each node display the prototype image patch and whether the image
matched (high similarity → went right) or did not match (low → went left).
Annotate with the routing probability. End at the leaf with the predicted class.

**Full tree figure (static):** only feasible for depth ≤ 4. Show all nodes as
image patches arranged as a binary tree. For depth 6+ only show the path for a
specific test image.

### What you need from CUB

Only `image_class_labels.txt` and `train_test_split.txt`.

---

## TesNet

**Paper:** Wang et al., ICCV 2021 — "Interpretable Image Recognition by
Constructing Transparent Embedding Space"

### Architecture

```
image → backbone → (B, 512, 7, 7) → global average pool → (B, 512)
      → project onto K orthogonal concept vectors → (B, K) concept scores
      → FC classifier → logits
```

The concept basis is a learnable matrix of shape `(K, 512)`. Concept scores
are the dot products of the pooled feature vector with each (normalised) basis
vector.

### Training

Loss = cross-entropy + orthogonality regularisation:

```python
L_ortho = MSE(basis_norm @ basis_norm.T, I_K)
L_total = L_cls + λ * L_ortho
```

The orthogonality loss keeps concepts distinct. As K increases, this
constraint becomes harder to satisfy in practice — a core limitation of TesNet.

There is **no discrete push step** during training. After training, concept
vectors are mapped to their nearest training patches purely for visualisation.

### `training_step` loss dict

```python
{"total": loss, "cls": l_cls, "ortho": l_ortho}
```

### `explain()` output

```python
{
    "concept_scores":   Tensor,  # (K,) — projection of this image onto each concept
    "top_concepts":     Tensor,  # indices of the top-K activated concepts
}
```

### Visualisation

**Concept activation bar chart:** horizontal bar chart of the K concept scores
for a given test image. Shows which concepts are active for this prediction.

**Concept visualisation (per concept):** for each concept, retrieve the
training image patches that maximally activate it (highest dot product with the
concept vector). Display a grid of these patches to give a human-interpretable
sense of what each concept represents.

Note: concepts are not guaranteed to be human-meaningful. A concept may
correspond to "texture in the upper-left region" rather than a bird part. This
is TesNet's key limitation and the main point to address in the Discussion
section of the report.

### What you need from CUB

Only `image_class_labels.txt` and `train_test_split.txt`.

---

## PIPNet

**Paper:** Nauta et al., CVPR 2023 — "PIP-Net: Patch-based Intuitive Prototypes"

### Architecture

```
image → backbone → (B, 512, 7, 7) → cosine similarity with prototypes
      → ReLU(similarity - threshold) → sparse scores → FC → logits
```

Prototypes are constrained to be actual training image patches during training
(not arbitrary latent vectors). The sparsity threshold means only 2–3
prototypes fire per image — the rest are exactly zero.

### Training

There is **no discrete push step** — the patch constraint is enforced through
the loss function during training. Prototypes are updated to remain close to
real training patches throughout.

Loss includes a sparsity term to encourage few prototypes to activate per image
and a cross-entropy classification term.

### `training_step` loss dict

```python
{"total": loss, "cls": l_cls, "sparsity": l_sparsity}
```

### `explain()` output

```python
{
    "active_prototypes":  Tensor,      # indices of prototypes that fired (sparse, ~2-3)
    "patch_locations":    Tensor,      # (n_active, 2) — (row, col) in 7×7 grid
    "activation_maps":    Tensor,      # (n_active, 7, 7)
    "prototype_sources":  list[Path],  # path to the training image each prototype came from
}
```

### Visualisation

The most direct of all four methods:

For each of the 2–3 active prototypes:
1. Upsample the `(7, 7)` activation map to `(224, 224)` and overlay on the
   test image as a heatmap (same as ProtoPNet).
2. Alongside, display the actual training image that the prototype IS (not
   merely similar to — it is that patch).

Because only 2–3 prototypes fire, the explanation is compact and readable.
The report figure for PIPNet should emphasise this sparsity relative to
ProtoPNet (which may have 50+ prototype scores to interpret).

### What you need from CUB

Only `image_class_labels.txt` and `train_test_split.txt`. The training image
paths stored in `prototype_sources` come from the Dataset's own `samples`
list — no additional annotation file is needed.

---

## Summary comparison

| Aspect | ProtoPNet | ProtoTree | TesNet | PIPNet |
|---|---|---|---|---|
| Prototype type | Latent vectors | Latent vectors | Concept directions | Real patches |
| Prototypes shared across classes | No | Yes | Yes | Yes |
| Discrete push step | Yes (Phase 2) | Yes | No | No |
| Explanation sparsity | Low (many scores) | Medium (one path) | Low (K scores) | High (2–3 active) |
| Explanation clarity | Moderate | Moderate | Low (abstract) | High |
| Multi-term loss | Yes (3 terms) | No / optional | Yes (2 terms) | Yes (2 terms) |

---

## Common visualisation utility — feature map to image overlay

All patch-based methods (ProtoPNet, ProtoTree, PIPNet) use the same core
routine to go from a `(7, 7)` activation map to an image-space heatmap:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

def overlay_activation(image: Image.Image, activation: torch.Tensor, alpha: float = 0.5):
    """
    image:      PIL Image, (224, 224, 3)
    activation: Tensor (7, 7) — raw activation for one prototype
    """
    act = activation.unsqueeze(0).unsqueeze(0).float()           # (1, 1, 7, 7)
    act = F.interpolate(act, size=(224, 224), mode="bilinear", align_corners=False)
    act = act.squeeze().numpy()
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)    # normalise to [0, 1]

    heatmap = plt.cm.hot(act)[:, :, :3]                          # (224, 224, 3)
    img_array = np.array(image) / 255.0
    blended = (1 - alpha) * img_array + alpha * heatmap

    return Image.fromarray((blended * 255).astype(np.uint8))
```

This utility belongs in a `src/visualize.py` module (to be implemented
alongside the notebooks) and is called from notebooks only — never during
training or evaluation.
