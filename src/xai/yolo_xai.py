from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class XAIResult:
    gradcam_overlay: np.ndarray | None = None
    feature_maps: list[tuple[str, np.ndarray]] = field(default_factory=list)
    target_layer_name: str = ""
    note: str = ""


class YOLOXAI:
    def __init__(self, model: Any, img_size: int = 640):
        self.yolo_model = model
        self.model = getattr(model, "model", model)
        self.img_size = img_size

    def _infer_device(self) -> torch.device:
        if not isinstance(self.model, nn.Module):
            return torch.device("cpu")
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _reset_inference_caches(self) -> None:
        if not isinstance(self.model, nn.Module):
            return

        for module in self.model.modules():
            for attr in ("anchors", "strides"):
                tensor = getattr(module, attr, None)
                if isinstance(tensor, torch.Tensor):
                    try:
                        if tensor.is_inference():
                            setattr(module, attr, tensor.clone())
                    except Exception:
                        # Older torch builds may not expose is_inference(); best-effort only.
                        pass
            # Force recomputation when shape-dependent caches exist.
            if hasattr(module, "shape"):
                try:
                    setattr(module, "shape", None)
                except Exception:
                    pass

    def generate(self, image: np.ndarray) -> XAIResult:
        if image is None or image.size == 0:
            return XAIResult(note="Empty image")

        if not isinstance(self.model, nn.Module):
            return XAIResult(note="Model is not a torch module")

        modules = self._module_list()
        target_layer = self._find_target_layer(modules)
        if target_layer is None:
            return XAIResult(note="Could not locate a target layer for Grad-CAM")

        feature_layers = self._find_feature_layers(modules)
        device = self._infer_device()
        input_tensor = self._prepare_tensor(image, device=device)

        # Give each layer a unique, stable name to avoid collisions (many layers share class names).
        name_map: dict[int, str] = {id(module): f"{idx}-{module.__class__.__name__}" for idx, module in enumerate(modules)}

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}
        handles = []

        def capture_forward(name: str):
            def hook(_module, _inputs, output):
                tensor = self._to_tensor(output)
                if tensor is not None:
                    activations[name] = tensor
            return hook

        def capture_backward(name: str):
            def hook(_module, _grad_inputs, grad_outputs):
                if grad_outputs:
                    tensor = self._to_tensor(grad_outputs[0])
                    if tensor is not None:
                        gradients[name] = tensor
            return hook

        target_name = name_map.get(id(target_layer), self._layer_name(target_layer))
        handles.append(target_layer.register_forward_hook(capture_forward(target_name)))
        handles.append(target_layer.register_full_backward_hook(capture_backward(target_name)))

        for layer in feature_layers:
            layer_name = name_map.get(id(layer), self._layer_name(layer))
            handles.append(layer.register_forward_hook(capture_forward(layer_name)))

        self.model.zero_grad(set_to_none=True)
        self.model.eval()

        # Ensure cached inference-mode tensors do not break Grad-CAM autograd.
        self._reset_inference_caches()

        with torch.enable_grad():
            outputs = self.model(input_tensor)
            score = self._select_score(outputs)
            if score is None:
                self._cleanup_handles(handles)
                return XAIResult(note="Could not derive a scalar score from model outputs")
            score.backward(retain_graph=False)

        gradcam = self._build_gradcam(
            activations.get(target_name),
            gradients.get(target_name),
            image.shape[:2],
        )

        feature_maps: list[tuple[str, np.ndarray]] = []
        for name, tensor in activations.items():
            if name == target_name:
                continue
            feature_maps.append((name, self._tensor_to_feature_map(tensor)))

        self._cleanup_handles(handles)

        return XAIResult(
            gradcam_overlay=self._overlay_heatmap(image, gradcam) if gradcam is not None else None,
            feature_maps=feature_maps,
            target_layer_name=target_name,
            note="OK",
        )

    def _prepare_tensor(self, image: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
        resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor.requires_grad_(True)

    def _select_score(self, outputs: Any) -> torch.Tensor | None:
        tensor = self._to_tensor(outputs)
        if tensor is not None:
            return tensor.max()

        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                tensor = self._to_tensor(item)
                if tensor is not None:
                    return tensor.max()

        return None

    def _find_target_layer(self, modules: list[nn.Module] | None = None):
        if modules is None:
            modules = self._module_list()
        if not modules:
            return None

        non_detect = [module for module in modules if not self._is_detect_layer(module)]
        if non_detect:
            return non_detect[-1]
        return modules[-1]

    def _find_feature_layers(self, modules: list[nn.Module] | None = None) -> list[nn.Module]:
        if modules is None:
            modules = self._module_list()
        non_detect = [module for module in modules if not self._is_detect_layer(module)]
        return non_detect[-4:-1] if len(non_detect) >= 4 else non_detect[:-1]

    def _module_list(self) -> list[nn.Module]:
        if hasattr(self.model, "model"):
            inner = getattr(self.model, "model")
            if isinstance(inner, (nn.Sequential, nn.ModuleList)):
                return list(inner)
            if isinstance(inner, nn.Module):
                return list(inner.children())
        return list(self.model.children())

    def _is_detect_layer(self, module: nn.Module) -> bool:
        return module.__class__.__name__.lower() == "detect"

    def _layer_name(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def _to_tensor(self, value: Any) -> torch.Tensor | None:
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = self._to_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _build_gradcam(
        self,
        activation: torch.Tensor | None,
        gradient: torch.Tensor | None,
        output_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if activation is None or gradient is None:
            return None

        if activation.ndim == 4:
            activation = activation[0]
        if gradient.ndim == 4:
            gradient = gradient[0]

        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation).sum(dim=0)
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()
        cam_np = cv2.resize(cam_np, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_LINEAR)
        return cam_np

    def _overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        heatmap_u8 = np.uint8(np.clip(heatmap * 255.0, 0, 255))
        color_map = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.55, color_map, 0.45, 0)
        return overlay

    def _tensor_to_feature_map(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor.mean(dim=0)

        fmap = tensor.detach().cpu().numpy()
        fmap = fmap - fmap.min()
        if fmap.max() > 0:
            fmap = fmap / fmap.max()
        return np.uint8(np.clip(fmap * 255.0, 0, 255))

    def _cleanup_handles(self, handles: list) -> None:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass