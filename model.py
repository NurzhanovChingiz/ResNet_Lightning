"""Residual network blocks and ImageNet-depth ResNet variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """Return a 3x3 convolution with padding.

    Args:
        in_planes: Input channel count.
        out_planes: Output channel count.
        stride: Convolution stride.
        groups: Grouped-convolution group count.
        dilation: Kernel dilation, also used as padding.

    Returns:
        Bias-free 3x3 ``Conv2d``.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Return a 1x1 convolution.

    Args:
        in_planes: Input channel count.
        out_planes: Output channel count.
        stride: Convolution stride.

    Returns:
        Bias-free 1x1 ``Conv2d``.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two-convolution residual block with expansion 1."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        *,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Store the two 3x3 convolutions and optional shortcut projection.

        Args:
            inplanes: Input channel count.
            planes: Output channel count before expansion.
            stride: Spatial stride of the first convolution.
            downsample: Optional projection for the identity shortcut.
            groups: Must be 1 for this block.
            base_width: Must be 64 for this block.
            dilation: Must be 1 for this block.
            norm_layer: Normalization constructor; BatchNorm2d when omitted.

        Returns:
            None.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            msg = "BasicBlock only supports groups=1 and base_width=64"
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 not supported in BasicBlock"
            raise NotImplementedError(msg)
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Apply a two-convolution residual unit.

        Purpose:
            Transform ``x`` through two 3x3 convolutions and add an aligned
            shortcut so the block can learn a residual.

        Strategy:
            Keep the input as identity, run the residual branch, then add a
            downsampled shortcut when spatial or channel size changed.

        Args:
            x: Input feature map.

        Returns:
            Residual-block output after the post-addition ReLU.

        Algorithm:
            1. Preserve the identity shortcut
            2. Compute the two-convolution residual branch
            3. Add the aligned shortcut and activate
        """
        # 1. Preserve the identity shortcut
        identity = x

        # 2. Compute the two-convolution residual branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 3. Add the aligned shortcut and activate
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return cast("Tensor", self.relu(out))


class Bottleneck(nn.Module):
    """1x1-3x3-1x1 residual block with expansion 4 (ResNet V1.5 stride)."""

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        *,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Store the bottleneck convolutions and optional shortcut projection.

        Args:
            inplanes: Input channel count.
            planes: Base channel width before expansion.
            stride: Spatial stride of the 3x3 convolution.
            downsample: Optional projection for the identity shortcut.
            groups: Grouped-convolution group count.
            base_width: Width scaling used with ``groups``.
            dilation: Dilation of the 3x3 convolution.
            norm_layer: Normalization constructor; BatchNorm2d when omitted.

        Returns:
            None.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Apply a bottleneck residual unit.

        Purpose:
            Transform ``x`` through a 1x1-3x3-1x1 branch and add an aligned
            shortcut so the block can learn a residual.

        Strategy:
            Keep the input as identity, run the bottleneck branch, then add a
            downsampled shortcut when spatial or channel size changed.

        Args:
            x: Input feature map.

        Returns:
            Residual-block output after the post-addition ReLU.

        Algorithm:
            1. Preserve the identity shortcut
            2. Compute the 1x1-3x3-1x1 residual branch
            3. Add the aligned shortcut and activate
        """
        # 1. Preserve the identity shortcut
        identity = x

        # 2. Compute the 1x1-3x3-1x1 residual branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 3. Add the aligned shortcut and activate
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return cast("Tensor", self.relu(out))


class ResNet(nn.Module):
    """Residual network stem, four stages, and a linear classification head."""

    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        *,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Build the stem, residual stages, and classification head.

        Args:
            block: Residual block class used in every stage.
            layers: Block counts for the four stages.
            num_classes: Linear-head output size.
            zero_init_residual: Zero the last BN scale in each block.
            groups: Grouped-convolution group count for bottleneck blocks.
            width_per_group: Per-group width used with ``groups``.
            replace_stride_with_dilation: Optional three flags replacing
                stride with dilation in stages 2-4.
            norm_layer: Normalization constructor; BatchNorm2d when omitted.

        Returns:
            None.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            msg = (
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
            raise ValueError(msg)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.initialize_weights(zero_init_residual=zero_init_residual)

    def initialize_weights(self, *, zero_init_residual: bool) -> None:
        """Initialize convolution, norm, and optional residual-branch scales.

        Args:
            zero_init_residual: Zero the last BN scale in each residual block.

        Returns:
            None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        *,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Build one residual stage as a sequential stack of blocks.

        Purpose:
            Produce ``blocks`` residual units at ``planes`` width, downsampling
            only on the first block when stride or channel count requires it.

        Strategy:
            Convert stride into dilation when requested, attach a 1x1
            projection if shapes would not add, then stack the first
            stride-bearing block followed by identity-stride blocks.

        Args:
            block: Residual block class for this stage.
            planes: Base channel width before ``block.expansion``.
            blocks: Number of residual blocks in the stage.
            stride: Spatial stride of the first block.
            dilate: When True, multiply stored dilation by ``stride`` and
                keep the first block at stride 1.

        Returns:
            Sequential module containing the stage's residual blocks.

        Flow:
            1. Resolve stride and dilation
            2. Build downsample when shapes differ
            3. Append the first block at this stride
            4. Append remaining matching-width blocks
            5. Return the sequential layer
        """
        # 1. Resolve stride and dilation
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 2. Build downsample when shapes differ
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 3. Append the first block at this stride
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        # 4. Append remaining matching-width blocks
        layers.extend(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )
            for _ in range(1, blocks)
        )

        # 5. Return the sequential layer
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Map a batched image tensor to class logits.

        Purpose:
            Run the ResNet stem, residual stages, and classification head.

        Strategy:
            Downsample through the stem, apply four residual stages, then
            globally pool and classify. This body is the torch.jit-friendly
            implementation behind ``forward``.

        Args:
            x: Batched input images.

        Returns:
            Class logits for each item in the batch.

        Flow:
            1. Apply stem convolution and pooling
            2. Apply residual stages
            3. Pool, flatten, and classify
        """
        # 1. Apply stem convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2. Apply residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 3. Pool, flatten, and classify
        x = self.avgpool(x)
        x = x.flatten(1)
        return cast("Tensor", self.fc(x))

    def forward(self, x: Tensor) -> Tensor:
        """Run the torch.jit-friendly residual network.

        Args:
            x: Batched input images.

        Returns:
            Class logits for each item in the batch.
        """
        return self._forward_impl(x)

    def _load_pretrained_weights(self, checkpoint_url: str, num_classes: int) -> None:
        """Load overlapping tensors from a checkpoint URL, then retarget the head.

        Purpose:
            Initialize this network from a public ResNet checkpoint without
            requiring an exact key and class-count match.

        Strategy:
            Download the state dict, keep keys this module already owns, load
            them, then replace ``self.classifier`` when the class count is not
            ImageNet-1000.

        Args:
            checkpoint_url: HTTPS URL of a torchvision ResNet ``.pth`` file.
            num_classes: Requested output class count.

        Returns:
            None. Mutates this module in place.

        Flow:
            1. Snapshot this module's state dict
            2. Download the checkpoint tensors
            3. Keep overlapping keys and apply them
            4. Replace the classifier when the class count is not 1000
        """
        # 1. Snapshot this module's state dict
        state_dict = nn.Module.state_dict(self)
        # 2. Download the checkpoint tensors
        pretrained_dict = torch.hub.load_state_dict_from_url(
            checkpoint_url,
            progress=True,
        )
        # 3. Keep overlapping keys and apply them
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
        state_dict.update(pretrained_dict)
        nn.Module.load_state_dict(self, state_dict)
        # 4. Replace the classifier when the class count is not 1000
        if num_classes != 1000:
            # nn.Module.__getattr__ types undeclared attributes as Tensor | Module.
            classifier = cast("Any", self.classifier)
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)


class ResNet18(ResNet):
    """ResNet-18 with BasicBlock stages ``[2, 2, 2, 2]``."""

    def __init__(self, num_classes: int = 1000, *, pretrained: bool = False) -> None:
        """Build ResNet-18 and optionally load a torchvision checkpoint.

        Args:
            num_classes: Output class count.
            pretrained: When True, load overlapping ImageNet tensors.

        Returns:
            None.
        """
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        if pretrained:
            self._load_pretrained_weights(
                "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                num_classes,
            )


class ResNet34(ResNet):
    """ResNet-34 with BasicBlock stages ``[3, 4, 6, 3]``."""

    def __init__(self, num_classes: int = 1000, *, pretrained: bool = False) -> None:
        """Build ResNet-34 and optionally load a torchvision checkpoint.

        Args:
            num_classes: Output class count.
            pretrained: When True, load overlapping ImageNet tensors.

        Returns:
            None.
        """
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
        if pretrained:
            self._load_pretrained_weights(
                "https://download.pytorch.org/models/resnet34-b627a593.pth",
                num_classes,
            )


class ResNet50(ResNet):
    """ResNet-50 with Bottleneck stages ``[3, 4, 6, 3]``."""

    def __init__(self, num_classes: int = 1000, *, pretrained: bool = False) -> None:
        """Build ResNet-50 and optionally load a torchvision checkpoint.

        Args:
            num_classes: Output class count.
            pretrained: When True, load overlapping ImageNet tensors.

        Returns:
            None.
        """
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
        if pretrained:
            self._load_pretrained_weights(
                "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                num_classes,
            )


class ResNet101(ResNet):
    """ResNet-101 with Bottleneck stages ``[3, 4, 23, 3]``."""

    def __init__(self, num_classes: int = 1000, *, pretrained: bool = False) -> None:
        """Build ResNet-101 and optionally load a torchvision checkpoint.

        Args:
            num_classes: Output class count.
            pretrained: When True, load overlapping ImageNet tensors.

        Returns:
            None.
        """
        super().__init__(
            block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes
        )
        if pretrained:
            self._load_pretrained_weights(
                "https://download.pytorch.org/models/resnet101-63fe2227.pth",
                num_classes,
            )


class ResNet152(ResNet):
    """ResNet-152 with Bottleneck stages ``[3, 8, 36, 3]``."""

    def __init__(self, num_classes: int = 1000, *, pretrained: bool = False) -> None:
        """Build ResNet-152 and optionally load a torchvision checkpoint.

        Args:
            num_classes: Output class count.
            pretrained: When True, load overlapping ImageNet tensors.

        Returns:
            None.
        """
        super().__init__(
            block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes
        )
        if pretrained:
            self._load_pretrained_weights(
                "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
                num_classes,
            )
