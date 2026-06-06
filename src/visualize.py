"""Prototype visualisation utilities for ProtoPNet (notebook-only).

These helpers are imported from notebooks for qualitative figures — they must
never run during training or evaluation. The core ``overlay_activation``
routine (feature map -> image-space heatmap) follows the shared recipe in
``docs/PROTOTYPE_METHODS_DETAILS.md`` and is reusable by the other patch-based methods.

The headline figure is the paper's "this looks like that": for each activated
prototype we show the **test** image with the prototype's heatmap + a bounding
box around its high-activation region, paired with the **source training image**
the prototype was pushed to, boxed around the same prototype's region there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

# matplotlib is an optional (notebook) dependency; import lazily so the module
# can be imported in headless test environments without it.
try:  # pragma: no cover - exercised only in notebooks
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

# Activation overlay blend weights — the original local_analysis recipe is
# ``0.5 * image + 0.3 * heatmap`` (a deliberately darkened overlay).
IMG_WEIGHT = 0.5
HEAT_WEIGHT = 0.3
# Percentile of the upsampled activation kept when drawing the bounding box
# (Chen et al. ``find_high_activation_crop`` default).
BBOX_PERCENTILE = 95.0
# A 95th-percentile high-activation region wider than this fraction of the image
# is treated as diffuse/saturated (no localised peak) — see :func:`_proto_bbox`.
DIFFUSE_AREA_FRAC = 0.5
# Receptive-field fallback box size, in feature-cell units: the box side spans
# ~RF_CELLS cells centred on the most-activated cell. Approximates the effective
# receptive field of one 1x1 prototype unit on the 7x7 grid of a 224px input.
RF_CELLS = 3.0
TEST_BOX_COLOR = (0, 255, 255)  # cyan
SOURCE_BOX_COLOR = (255, 255, 0)  # yellow


def paper_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Deterministic, paper-faithful eval/push transform (square resize, no crop).

    Chen et al. (``main.py``) resize **every** split — train, push and test — with
    ``Resize((img_size, img_size))`` (a direct square resize, *not*
    ``Resize(256) + CenterCrop(224)``). Reusing the exact same recipe for the
    push loader, the eval loaders and the source-image rendering here keeps three
    things aligned: (1) train and eval see the same scale/aspect, (2) the prototype
    figures match the scale the paper's figures use, and (3) the source image is
    framed at push time exactly as it is re-rendered at visualisation time, so the
    7x7 grid / heatmap / box line up. Import this in the notebook so the data
    loaders and the visualiser cannot drift apart.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(image: torch.Tensor) -> Image.Image:
    """Undo ImageNet normalisation on a ``(3, H, W)`` tensor -> PIL image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    array = (image.detach().cpu() * std + mean).clamp(0, 1)
    array = (array.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def _upsample_activation(activation: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Cubic-resize a ``(H, W)`` activation to image size and normalise to [0, 1].

    Mirrors the original ``cv2.resize(..., interpolation=cv2.INTER_CUBIC)`` step;
    ``bicubic`` is torch's nearest equivalent. Bicubic can overshoot [0, 1], so
    the min-max normalisation afterwards re-clamps the range.
    """
    act = activation.detach().cpu().float().view(1, 1, *activation.shape)
    act = F.interpolate(act, size=size, mode="bicubic", align_corners=False)
    act = act.squeeze().numpy()
    return (act - act.min()) / (act.max() - act.min() + 1e-8)


def overlay_activation(
    image: Image.Image,
    activation: torch.Tensor,
    img_weight: float = IMG_WEIGHT,
    heat_weight: float = HEAT_WEIGHT,
) -> Image.Image:
    """Overlay a prototype activation map on an image as a JET heatmap.

    Follows the original ``local_analysis`` recipe: cubic-upsample the activation
    to image size, min-max normalise, colour it with the ``jet`` colormap and
    blend ``img_weight * image + heat_weight * heatmap``.

    :param image: PIL image (any size; the activation is resized to match).
    :param activation: raw activation for one prototype, shape ``(H, W)``.
    """
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required for overlay_activation()")

    act = _upsample_activation(activation, (image.height, image.width))
    heatmap = plt.cm.jet(act)[:, :, :3]
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    blended = img_weight * img_array + heat_weight * heatmap
    return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8))


def activation_bbox(
    activation: torch.Tensor, size: tuple[int, int], percentile: float = BBOX_PERCENTILE
) -> tuple[int, int, int, int] | None:
    """Bounding box ``(x0, y0, x1, y1)`` around the high-activation region.

    The activation is upsampled to ``size`` and thresholded at ``percentile``;
    the box is the extent of the surviving mask (the paper's prototype region).
    """
    act = _upsample_activation(activation, size)
    threshold = np.percentile(act, percentile)
    rows, cols = np.where(act >= threshold)
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def draw_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int] = TEST_BOX_COLOR,
    width: int = 3,
) -> Image.Image:
    """Return a copy of ``image`` with ``bbox`` drawn (no-op if bbox is None)."""
    if bbox is None:
        return image
    out = image.copy()
    ImageDraw.Draw(out).rectangle(bbox, outline=color, width=width)
    return out


def _resolve_source(
    push_dataset, index: int
) -> tuple[Path | None, tuple | None]:
    """Map a prototype's source index to its training image ``(path, bbox)``.

    Handles both a raw ``CUB200Dataset`` (has ``.samples``) and a ``Subset``
    (has ``.indices`` + ``.dataset``). The index space must match the dataset
    that produced the (unshuffled) push loader. Each ``samples`` entry is
    ``(path, label, bbox)`` (``bbox`` is ``None`` when bbox-crop is off); the
    bbox MUST be applied at visualisation time so the source image is framed
    exactly as it was when the prototype was pushed (otherwise the 7x7 grid no
    longer lines up and the heatmap/box land on the wrong region).
    """
    if hasattr(push_dataset, "samples"):
        samples = push_dataset.samples
        if index >= len(samples):
            return None, None
        record = samples[index]
        return Path(record[0]), (record[2] if len(record) > 2 else None)
    if hasattr(push_dataset, "indices") and hasattr(push_dataset, "dataset"):
        if index >= len(push_dataset.indices):
            return None, None
        record = push_dataset.dataset.samples[push_dataset.indices[index]]
        return Path(record[0]), (record[2] if len(record) > 2 else None)
    return None, None


def _prepare_image(
    pil_image: Image.Image, image_size: int, device, bbox: tuple | None = None
) -> torch.Tensor:
    """Deterministic (no-augmentation) transform -> ``(1, 3, H, W)`` tensor.

    Mirrors ``CUB200Dataset.__getitem__``: crop to the bird bounding box on the
    raw PIL image *before* the resize/normalise transform, so the framing
    matches what the backbone saw during the push step.
    """
    if bbox is not None:
        x, y, w, h = bbox
        pil_image = pil_image.crop((x, y, x + w, y + h))
    # Square resize (no center-crop): MUST match the push loader's transform so
    # the re-rendered source frames the bird exactly as it was at push time.
    transform = paper_eval_transform(image_size)
    return transform(pil_image).unsqueeze(0).to(device)


def cell_bbox(
    location: tuple[int, int] | torch.Tensor,
    grid_size: int,
    image_size: int,
    expand: float = 0.5,
) -> tuple[int, int, int, int]:
    """Pixel box for a prototype's most-activated grid cell.

    The prototype layer matches a single cell of the ``grid_size`` x
    ``grid_size`` feature map. Mapping that cell back to image space gives a
    tight, deterministic box that — unlike percentile-thresholding an upsampled
    heatmap — stays meaningful even when the activation map is nearly flat
    (saturated similarities). ``expand`` widens the box to approximate the
    cell's receptive field (0.5 -> roughly double the cell side).
    """
    row, col = int(location[0]), int(location[1])
    cell = image_size / grid_size
    center_x = (col + 0.5) * cell
    center_y = (row + 0.5) * cell
    half = cell * (0.5 + expand)
    x0 = max(0, int(round(center_x - half)))
    y0 = max(0, int(round(center_y - half)))
    x1 = min(image_size, int(round(center_x + half)))
    y1 = min(image_size, int(round(center_y + half)))
    return x0, y0, x1, y1


def _proto_bbox(
    activation: torch.Tensor,
    size: tuple[int, int],
    location=None,
    grid_size: int | None = None,
) -> tuple[int, int, int, int] | None:
    """Paper-faithful prototype box, robust to saturated activation maps.

    Primary: the 95th-percentile high-activation region of the cubic-upsampled
    map (``activation_bbox`` == the original ``find_high_activation_crop``). This
    is the box the paper draws in its "this looks like that" figures.

    Fallback: a weakly-trained prototype can produce a flat/saturated similarity
    map whose 95th-percentile pixels are scattered by noise, so the enclosing
    rectangle degenerates into an arbitrary box covering much of the image (or is
    empty). In that case — an empty box, or one wider than ``DIFFUSE_AREA_FRAC``
    of the image — we instead draw a receptive-field box centred on the
    most-activated grid cell (``RF_CELLS`` cells wide): a stable, localised region
    rather than a noise-driven one. The deeper cure for diffuse maps is a
    well-clustered latent space (the iterative push schedule), after which the
    primary percentile box is used as normal.
    """
    box = activation_bbox(activation, size, BBOX_PERCENTILE)
    height, width = size
    diffuse = box is not None and (
        (box[2] - box[0]) * (box[3] - box[1]) > DIFFUSE_AREA_FRAC * width * height
    )
    if (box is None or diffuse) and location is not None and grid_size is not None:
        return cell_bbox(location, grid_size, size[0], expand=(RF_CELLS - 1.0) / 2.0)
    return box


def _rank_prototypes(model, similarities: torch.Tensor, predicted: int, top_k: int):
    """Indices of the ``top_k`` prototypes that most drive the prediction.

    Ranks the **predicted class's own** prototypes by their contribution to that
    class's logit (``similarity * classifier weight``) — this matches the
    paper's figure, which explains a decision with the predicted class's
    prototypes, not whatever patch is globally closest (which is usually an
    unrelated class and looks nonsensical).
    """
    class_mask = model.prototype_class_identity[:, predicted].bool()  # (P,)
    weights = model.classifier.weight[predicted]  # (P,)
    contribution = similarities * weights
    contribution = contribution.masked_fill(~class_mask, float("-inf"))
    k = min(top_k, int(class_mask.sum().item()))
    return contribution.topk(k).indices.tolist()


def _render_prototype_row(
    ax_test,
    ax_source,
    model,
    proto: int,
    *,
    test_base: Image.Image,
    test_maps: torch.Tensor,
    test_locations: torch.Tensor,
    similarities: torch.Tensor,
    predicted: int,
    grid_size: int,
    test_px: int,
    sources,
    push_dataset,
    device,
    image_size: int,
    title_prefix: str = "",
) -> None:
    """Render one "this looks like that" row (test panel + source panel).

    Shared by both public figures. The box on each panel is the prototype's
    high-activation region (``_proto_bbox``); the source panel re-frames the
    training image with the SAME bbox crop used at push time so its 7x7 grid
    lines up with the cached source location.
    """
    cls = int(model.prototype_class_identity[proto].argmax())

    # --- left: test image ("this") ---
    test_overlay = overlay_activation(test_base, test_maps[proto])
    test_overlay = draw_bbox(
        test_overlay,
        _proto_bbox(test_maps[proto], (test_px, test_px), test_locations[proto], grid_size),
        TEST_BOX_COLOR,
    )
    ax_test.imshow(test_overlay)
    ax_test.set_title(
        f"{title_prefix}This (test) · pred class {predicted}\n"
        f"prototype {proto} · class {cls} · sim={similarities[proto]:.2f}"
    )
    ax_test.axis("off")

    # --- right: source training image ("that") ---
    source = sources[proto] if proto < len(sources) else None
    source_path, source_crop = (
        _resolve_source(push_dataset, source[0])
        if source is not None and push_dataset is not None
        else (None, None)
    )
    if source_path is not None and source_path.exists():
        source_pil = Image.open(source_path).convert("RGB")
        source_tensor = _prepare_image(source_pil, image_size, device, source_crop)
        source_explain = model.explain(source_tensor)
        source_map = source_explain["activation_maps"][0, proto]
        source_loc = source_explain["patch_locations"][0, proto]
        source_base = denormalize(source_tensor[0])
        source_overlay = overlay_activation(source_base, source_map)
        source_overlay = draw_bbox(
            source_overlay,
            _proto_bbox(
                source_map, (source_base.height, source_base.width), source_loc, grid_size
            ),
            SOURCE_BOX_COLOR,
        )
        ax_source.imshow(source_overlay)
        ax_source.set_title("...looks like that (source patch)")
    else:
        ax_source.text(
            0.5, 0.5, "no source\n(run push step\n+ pass push_dataset)",
            ha="center", va="center",
        )
    ax_source.axis("off")


def _explain_test_image(model, image: torch.Tensor, device):
    """Run a single test image through the model once, returning the pieces the
    figures need: predicted class + per-prototype similarities/maps/locations."""
    model.eval()
    model.to(device)
    test_batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = int(model(test_batch).argmax(dim=1).item())
    test_explain = model.explain(test_batch)
    return (
        predicted,
        test_explain["prototype_similarities"][0],  # (P,)
        test_explain["activation_maps"][0],         # (P, H, W)
        test_explain["patch_locations"][0],         # (P, 2)
    )


def _render_rows(model, image, protos, push_dataset, device, image_size,
                 predicted, similarities, test_maps, test_locations, title_prefix=""):
    """Build the 2-column figure for a list of prototype indices."""
    grid_size = test_maps.shape[-1]
    test_base = denormalize(image)
    test_px = test_base.height  # square after the test transform
    sources = getattr(model, "prototype_source_info", [None] * model.num_prototypes)

    fig, axes = plt.subplots(len(protos), 2, figsize=(8, 4 * len(protos)))
    if len(protos) == 1:
        axes = axes.reshape(1, 2)

    for row, proto in enumerate(protos):
        prefix = f"#{row + 1} · {title_prefix}" if title_prefix else ""
        _render_prototype_row(
            axes[row, 0], axes[row, 1], model, proto,
            test_base=test_base, test_maps=test_maps, test_locations=test_locations,
            similarities=similarities, predicted=predicted, grid_size=grid_size,
            test_px=test_px, sources=sources, push_dataset=push_dataset,
            device=device, image_size=image_size, title_prefix=prefix,
        )
    fig.tight_layout()
    return fig


def visualize_prototype_explanation(
    model,
    image: torch.Tensor,
    push_dataset=None,
    top_k: int = 3,
    device: str | torch.device = "cpu",
    image_size: int = 224,
):
    """Paper-style "this looks like that" figure for one test image.

    The prototypes shown are the **predicted class's** top contributors (ranked
    by ``similarity * classifier weight``). For each, two panels:

    * **left (test)** — the test image, prototype heatmap, cyan box on the
      high-activation region ("this");
    * **right (source)** — the training image the prototype was pushed to (framed
      with the SAME bbox crop used at push time), the prototype's heatmap
      re-computed on it, yellow box on its region ("that").

    Boxes use the paper's ``find_high_activation_crop`` (95th percentile of the
    cubic-upsampled activation). NOTE: when the dataset was loaded with
    ``use_bbox_crop=True`` the panels are framed to the bird's bounding box —
    this is **correct** and matches the original, which trains/visualises on the
    pre-cropped ``cub200_cropped`` images.

    :param model: a trained ProtoPNet, ideally **already pushed**.
    :param image: a single ``(3, H, W)`` normalised test image tensor.
    :param push_dataset: the (unshuffled) dataset used for the push step, so the
        cached source indices resolve to the right files. May be a ``Subset``.
    :returns: the matplotlib Figure.
    """
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required for visualize_prototype_explanation()")

    predicted, similarities, test_maps, test_locations = _explain_test_image(
        model, image, device
    )
    top_protos = _rank_prototypes(model, similarities, predicted, top_k)
    return _render_rows(
        model, image, top_protos, push_dataset, device, image_size,
        predicted, similarities, test_maps, test_locations,
    )


def visualize_most_activated_prototypes(
    model,
    image: torch.Tensor,
    push_dataset=None,
    top_k: int = 10,
    device: str | torch.device = "cpu",
    image_size: int = 224,
):
    """The paper's "most activated N prototypes" figure (``local_analysis.py``).

    Unlike :func:`visualize_prototype_explanation` (which restricts to the
    predicted class's prototypes), this ranks **all** prototypes by raw
    similarity to the test image — the globally nearest patches, regardless of
    class. It's a sanity check on what the latent space considers closest, and
    on a healthy model the top hits should belong to the predicted class.

    Same two-panel layout and box recipe as the per-class figure.
    """
    if plt is None:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for visualize_most_activated_prototypes()"
        )

    predicted, similarities, test_maps, test_locations = _explain_test_image(
        model, image, device
    )
    k = min(top_k, similarities.numel())
    top_protos = similarities.topk(k).indices.tolist()
    return _render_rows(
        model, image, top_protos, push_dataset, device, image_size,
        predicted, similarities, test_maps, test_locations, title_prefix="global",
    )
