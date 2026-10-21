"""Contracts for ResNet construction and pretrained weight loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn

from model import ResNet18, ResNet50

if TYPE_CHECKING:
    from collections.abc import Callable


def _imagenet_checkpoint(template: nn.Module) -> dict[str, Tensor]:
    """Return a cloned ImageNet-shaped state dict with marked tensors.

    Args:
        template: Network whose current tensors define checkpoint keys.

    Returns:
        Detached clones with ``conv1.weight`` filled with 0.25 and
        ``fc`` tensors filled with 0.75.
    """
    checkpoint = {
        key: tensor.detach().clone() for key, tensor in template.state_dict().items()
    }
    checkpoint["conv1.weight"].fill_(0.25)
    checkpoint["fc.weight"].fill_(0.75)
    checkpoint["fc.bias"].fill_(0.75)
    return checkpoint


def _stem_conv(model: nn.Module) -> nn.Conv2d:
    """Return the stem convolution after checking its type.

    Args:
        model: ResNet module that stores ``conv1``.

    Returns:
        The stem ``Conv2d``.
    """
    conv1 = model.conv1
    assert isinstance(conv1, nn.Conv2d)
    return conv1


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch, checkpoint: dict[str, Tensor]
) -> list[str]:
    """Patch ``torch.hub.load_state_dict_from_url`` to return ``checkpoint``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        checkpoint: Fake tensors returned instead of a download.

    Returns:
        List that receives each requested URL.
    """
    requested_urls: list[str] = []

    def fake_load_state_dict_from_url(
        url: str,
        *args: object,
        **kwargs: object,
    ) -> dict[str, Tensor]:
        del args, kwargs
        requested_urls.append(url)
        return checkpoint

    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url
    )
    return requested_urls


def test_pretrained_false_does_not_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager construction must not call torch hub."""

    def fail_download(*args: object, **kwargs: object) -> dict[str, Tensor]:
        del args, kwargs
        message = "torch.hub.load_state_dict_from_url must not run"
        raise AssertionError(message)

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fail_download)
    model = ResNet18(num_classes=10, pretrained=False)
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 10
    assert not hasattr(model, "classifier")


@pytest.mark.parametrize(
    ("builder", "num_classes", "in_features"),
    [
        (ResNet18, 10, 512),
        (ResNet50, 10, 2048),
    ],
)
def test_pretrained_transfer_loads_backbone_and_keeps_target_fc(
    monkeypatch: pytest.MonkeyPatch,
    builder: Callable[..., nn.Module],
    num_classes: int,
    in_features: int,
) -> None:
    """Transfer loading skips mismatched fc tensors and keeps a Linear head."""
    template = builder(num_classes=1000, pretrained=False)
    checkpoint = _imagenet_checkpoint(template)
    requested_urls = _install_fake_hub(monkeypatch, checkpoint)

    model = builder(num_classes=num_classes, pretrained=True)

    assert requested_urls
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == num_classes
    assert model.fc.in_features == in_features
    assert not hasattr(model, "classifier")
    conv1 = _stem_conv(model)
    assert torch.equal(conv1.weight, torch.full_like(conv1.weight, 0.25))
    assert model.fc.weight.shape == (num_classes, in_features)
    assert not torch.equal(
        model.fc.weight, torch.full((num_classes, in_features), 0.75)
    )
    logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, num_classes)


def test_pretrained_imagenet_loads_matching_fc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ImageNet-1000 loading applies same-shaped fc tensors."""
    template = ResNet18(num_classes=1000, pretrained=False)
    checkpoint = _imagenet_checkpoint(template)
    _install_fake_hub(monkeypatch, checkpoint)

    model = ResNet18(num_classes=1000, pretrained=True)

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 1000
    assert not hasattr(model, "classifier")
    assert torch.equal(model.fc.weight, checkpoint["fc.weight"])
    assert torch.equal(model.fc.bias, checkpoint["fc.bias"])
    conv1 = _stem_conv(model)
    assert torch.equal(conv1.weight, torch.full_like(conv1.weight, 0.25))
    logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 1000)
