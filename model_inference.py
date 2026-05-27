"""ResNet18 poultry disease inference — uses only ResNet18_poultry_diseases.pth."""

from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Same mapping as PoultryDataset in the notebook
LABEL_MAP = {"cocci": 0, "healthy": 1, "ncd": 2, "salmo": 3}
NUM_CLASSES = len(LABEL_MAP)
CLASS_NAMES = [name for name, _ in sorted(LABEL_MAP.items(), key=lambda x: x[1])]

CLASS_INFO = {
    "cocci": {
        "title": "Coccidiosis",
        "summary": "Intestinal parasite infection; common in crowded flocks.",
    },
    "healthy": {
        "title": "Healthy",
        "summary": "No obvious disease signs in the submitted image.",
    },
    "ncd": {
        "title": "Newcastle disease (NCD)",
        "summary": "Viral disease; respiratory and nervous signs may appear.",
    },
    "salmo": {
        "title": "Salmonella",
        "summary": "Bacterial infection; may affect gut and general condition.",
    },
}

MODEL_FILENAME = "ResNet18_poultry_diseases.pth"

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class ModelWeightsError(RuntimeError):
    """Raised when the poultry ResNet18 weights file is missing or invalid."""


def _validate_weights_file(path: Path) -> None:
    """
    Validate the model file before torch.load().
    This catches common deployment issues (text/LFS pointer/corrupt upload).
    """
    size = path.stat().st_size
    if size < 1024:
        raise ModelWeightsError(
            f"`{path.name}` looks too small ({size} bytes). "
            "Expected a binary PyTorch state_dict file."
        )

    header = path.read_bytes()[:16]
    # torch save commonly starts with ZIP header (PK...) or pickle protocol bytes.
    valid_binary = header.startswith(b"PK") or header.startswith(b"\x80")
    if not valid_binary:
        hex_head = header.hex()
        raise ModelWeightsError(
            f"`{path.name}` is not a valid PyTorch binary header (first bytes: {hex_head}). "
            "Re-upload the file in binary mode and ensure no line-ending/text conversion."
        )


def model_weights_path(base_dir: Path | None = None) -> Path:
    root = base_dir or Path(__file__).resolve().parent
    return root / MODEL_FILENAME


def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Initialize ResNet18 without ImageNet weights; replace head for 4 poultry classes."""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_poultry_resnet18(base_dir: Path | None = None) -> nn.Module:
    """Build ResNet18 (pretrained=False) and load weights from ResNet18_poultry_disease.pth."""
    path = model_weights_path(base_dir)
    if not path.is_file():
        raise ModelWeightsError(
            f"Required weights file not found: {path}\n"
            f"Place `{MODEL_FILENAME}` in the app folder."
        )
    _validate_weights_file(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18()

    # Prefer the safer weights_only=True path, but support older/legacy files.
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
        load_mode = "weights_only=True"
    except TypeError:
        # Older torch versions may not support weights_only.
        state_dict = torch.load(path, map_location=device)
        load_mode = "weights_only=False (torch compatibility fallback)"
    except Exception:
        # Some older .pth files are not compatible with weights_only=True.
        try:
            state_dict = torch.load(path, map_location=device, weights_only=False)
            load_mode = "weights_only=False (legacy file fallback)"
        except Exception as exc:
            raise ModelWeightsError(
                f"Failed to load `{path.name}` as a PyTorch state_dict: {exc}. "
                "This usually means the deployed file is corrupted or was transferred as text."
            ) from exc

    # If this is a training checkpoint, extract the actual model weights.
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    model._poultry_weights_source = str(path.resolve())  # type: ignore[attr-defined]
    model._weights_load_mode = load_mode  # type: ignore[attr-defined]
    return model


def load_model(base_dir: Path | None = None) -> tuple[nn.Module, str]:
    """Load poultry weights. Returns (model, status_message)."""
    path = model_weights_path(base_dir)
    model = load_poultry_resnet18(base_dir)
    device = next(model.parameters()).device
    return (
        model,
        f"ResNet18 (pretrained=False) loaded from {path.name} on {device} [{getattr(model, '_weights_load_mode', 'unknown mode')}].",
    )


def is_poultry_model_loaded(model: nn.Module) -> bool:
    return bool(getattr(model, "_poultry_weights_source", None))


def predict_image(model: nn.Module, image_bytes: bytes) -> dict:
    """Classify one image; model is set to eval() for deterministic inference."""
    if not is_poultry_model_loaded(model):
        raise ModelWeightsError(
            f"Inference blocked: load `{MODEL_FILENAME}` with load_poultry_resnet18() first."
        )

    device = next(model.parameters()).device
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    # eval() disables dropout / batch-norm training behavior for consistent predictions
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    prob_by_class = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": prob_by_class,
        "class_title": CLASS_INFO[pred_class]["title"],
        "class_summary": CLASS_INFO[pred_class]["summary"],
        "weights_file": Path(model._poultry_weights_source).name,  # type: ignore[attr-defined]
    }
