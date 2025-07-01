import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from model_utils import *
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from torch import Tensor

class PlainGRUTemporalProcessor(nn.Module):
    """Simple temporal processor using PyTorch's plain GRU (non-convolutional)"""
    
    def __init__(self, feature_channels=256, hidden_channels=128, num_layers=1, reduced_channels=1):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.reduced_channels = reduced_channels
        
        # 1x1 convolution to reduce to minimal channels (1 channel by default)
        self.conv_reduce = nn.Conv2d(feature_channels, reduced_channels, kernel_size=1)
        
        # Plain GRU for temporal processing (operates on flattened features)
        # Input size is reduced_channels * spatial_size after flattening
        self.gru = nn.GRU(
            input_size=reduced_channels,  # Will be multiplied by spatial size
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Feature modulation networks
        self.modulation_net = nn.Sequential(
            nn.Linear(hidden_channels, feature_channels),
            nn.Sigmoid()
        )
        
        self.hidden_state = None
        self.spatial_size = None  # Store spatial size for GRU input size
    
    def reset_state(self, batch_size, device):
        """Reset hidden state for new sequences"""
        self.hidden_state = None
        self.spatial_size = None
    
    def forward(self, features):
        """
        Apply temporal processing to features using plain GRU
        Args:
            features: OrderedDict of feature maps from FPN
        Returns:
            enhanced_features: OrderedDict with temporally enhanced features
        """
        if not features:
            return features
        
        # Work with the highest level features (most semantic)
        feature_key = '3' if '3' in features else list(features.keys())[-1]
        high_level_features = features[feature_key]
        
        batch_size = high_level_features.shape[0]
        device = high_level_features.device
        original_size = high_level_features.shape[2:]
        
        # Apply 1x1 convolution to reduce to minimal channels
        reduced_features = self.conv_reduce(high_level_features)  # (B, reduced_channels, H, W)
        
        # Flatten spatial dimensions for GRU processing
        spatial_size = reduced_features.shape[2] * reduced_features.shape[3]  # H * W
        flattened_features = reduced_features.view(batch_size, self.reduced_channels * spatial_size)  # (B, reduced_channels * H * W)
        
        # Reshape for GRU: (batch_size, seq_len=1, input_size)
        gru_input = flattened_features.unsqueeze(1)  # (B, 1, reduced_channels * H * W)
        
        # Update GRU input size if spatial dimensions changed
        if self.spatial_size != spatial_size:
            self.spatial_size = spatial_size
            # Reinitialize GRU with correct input size
            self.gru = nn.GRU(
                input_size=self.reduced_channels * spatial_size,
                hidden_size=self.hidden_channels,
                num_layers=self.num_layers,
                batch_first=True
            ).to(device)
            self.hidden_state = None
        
        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(
                self.num_layers, batch_size, self.hidden_channels, 
                device=device, dtype=gru_input.dtype
            )
        
        # GRU processing
        gru_output, self.hidden_state = self.gru(gru_input, self.hidden_state)
        
        # Extract output and generate modulation weights
        temporal_features = gru_output.squeeze(1)  # (B, hidden_channels)
        modulation = self.modulation_net(temporal_features)  # (B, feature_channels)
        
        # Reshape modulation to match original feature shape
        modulation = modulation.view(batch_size, self.feature_channels, 1, 1)
        
        # Upsample modulation to match original feature size
        if original_size != (1, 1):
            modulation = F.interpolate(
                modulation, size=original_size, 
                mode='bilinear', align_corners=False
            )
        
        # Apply temporal modulation to features
        enhanced_features = features.copy()
        enhanced_features[feature_key] = high_level_features * modulation
        
        return enhanced_features


class ConvGRUCell(nn.Module):
    """ConvGRU cell - simpler and more memory efficient alternative to ConvLSTM"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Reset and update gates
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,  # reset + update gates
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # New gate
        self.conv_new = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )

    def forward(self, input_tensor, hidden_state):
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        
        # Reset and update gates
        combined_conv = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        # New gate with reset applied
        reset_combined = torch.cat([input_tensor, reset_gate * hidden_state], dim=1)
        new_gate = torch.tanh(self.conv_new(reset_combined))
        
        # Update hidden state
        hidden_new = (1 - update_gate) * new_gate + update_gate * hidden_state
        
        return hidden_new

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)


class GlobalPoolingTemporalProcessorGRU(nn.Module):
    """Memory-efficient temporal processor using ConvGRU and global pooling"""
    
    def __init__(self, feature_channels=256, hidden_channels=128):
        super().__init__()
        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        
        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # ConvGRU for temporal processing (operates on 1x1 features)
        self.conv_gru = ConvGRUCell(
            input_dim=feature_channels,
            hidden_dim=hidden_channels,
            kernel_size=1  # 1x1 conv since we're working with pooled features
        )
        
        # Feature modulation networks
        self.modulation_net = nn.Sequential(
            nn.Conv2d(hidden_channels, feature_channels, 1),
            nn.Sigmoid()
        )
        
        self.hidden_state = None
    
    def reset_state(self, batch_size, device):
        """Reset hidden state for new sequences"""
        self.hidden_state = None
    
    def forward(self, features):
        """
        Apply temporal processing to features using ConvGRU
        Args:
            features: OrderedDict of feature maps from FPN
        Returns:
            enhanced_features: OrderedDict with temporally enhanced features
        """
        if not features:
            return features
        
        # Work with the highest level features (most semantic)
        feature_key = '3' if '3' in features else list(features.keys())[-1]
        high_level_features = features[feature_key]
        
        batch_size = high_level_features.shape[0]
        device = high_level_features.device
        original_size = high_level_features.shape[2:]
        
        # Global pooling to 1x1
        pooled_features = self.global_pool(high_level_features)  # (B, C, 1, 1)
        
        # Initialize hidden state if needed
        if self.hidden_state is None:
            self.hidden_state = self.conv_gru.init_hidden(
                batch_size, (1, 1), device
            )
        
        # ConvGRU processing (simpler than LSTM - only one state)
        self.hidden_state = self.conv_gru(pooled_features, self.hidden_state)
        
        # Generate modulation weights
        modulation = self.modulation_net(self.hidden_state)  # (B, C, 1, 1)
        
        # Upsample modulation to match original feature size
        if original_size != (1, 1):
            modulation = F.interpolate(
                modulation, size=original_size, 
                mode='bilinear', align_corners=False
            )
        
        # Apply temporal modulation to features
        enhanced_features = features.copy()
        enhanced_features[feature_key] = high_level_features * modulation
        
        return enhanced_features


def _default_anchorgen():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): Images are rescaled before feeding them to the backbone:
            we attempt to preserve the aspect ratio and scale the shorter edge
            to ``min_size``. If the resulting longer edge exceeds ``max_size``,
            then downscale so that the longer edge does not exceed ``max_size``.
            This may result in the shorter edge beeing lower than ``min_size``.
        max_size (int): See ``min_size``.
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): only return proposals with an objectness score greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280,
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(
        self,
        backbone,
        num_classes=None,
        # Temporal processing parameter (non-invasive addition)
        temporal_processor=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )
        if not isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"box_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(box_roi_pool)}"
            )

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        super().__init__(backbone, rpn, roi_heads, transform)
        
        # Add temporal processor (non-invasive)
        self.temporal_processor = temporal_processor

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor] or Tensor): images to be processed or sequence tensor
            targets (list[Dict[str, Tensor]] or list[list[Dict[str, Tensor]]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # Check if input is a sequence tensor (5D: batch, seq_len, channels, height, width)
        if isinstance(images, torch.Tensor) and len(images.shape) == 5:
            return self.forward_sequence(images, targets)
        
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # Apply temporal processing if available (non-invasive)
        if self.temporal_processor is not None:
            features = self.temporal_processor(features)

        proposals, proposal_losses = self.rpn(images, features, targets)
        box_features = self.roi_heads.box_roi_pool(features, proposals, images.image_sizes)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections

    def forward_sequence(self, sequences, targets_sequences=None):
        """
        Process a sequence of frames with temporal context
        
        Args:
            sequences: Tensor of shape (batch_size, seq_length, channels, height, width)
            targets_sequences: List of lists containing targets for each frame
        
        Returns:
            accumulated_losses (training) or final_detections (inference)
        """
        batch_size, seq_length = sequences.shape[:2]
        device = sequences.device
        
        # Reset temporal state for new sequences
        if self.temporal_processor:
            self.temporal_processor.reset_state(batch_size, device)
        
        accumulated_losses = {}
        final_detections = None
        
        for t in range(seq_length):
            frame = sequences[:, t]  # (batch_size, channels, height, width)
            frame_list = [frame[b] for b in range(batch_size)]
            
            # Extract targets for current frame
            if targets_sequences is not None:
                current_targets = [targets_sequences[b][t] for b in range(batch_size)]
            else:
                current_targets = None
            
            # Process frame with temporal enhancement
            frame_losses, frame_detections = self.forward(frame_list, current_targets)
            
            # Accumulate losses
            if self.training and current_targets is not None:
                for loss_name, loss_value in frame_losses.items():
                    if loss_name not in accumulated_losses:
                        accumulated_losses[loss_name] = loss_value
                    else:
                        accumulated_losses[loss_name] += loss_value
            
            # Keep final frame detections
            if t == seq_length - 1:
                final_detections = frame_detections
        
        # Average losses across sequence
        if self.training and accumulated_losses:
            for loss_name in accumulated_losses:
                accumulated_losses[loss_name] /= seq_length
            return accumulated_losses
        else:
            return final_detections

    def is_sequence_input(self, images):
        """Check if input is a sequence tensor"""
        return isinstance(images, torch.Tensor) and len(images.shape) == 5

def construct_model():
    trainable_backbone_layers = 5
    trainable_backbone_layers = _validate_trainable_layers(True, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    return FasterRCNN(backbone, num_classes=5)


def construct_temporal_model_plain_gru(num_classes=5, hidden_channels=128):
    """
    Construct a FasterRCNN model with plain GRU temporal processing
    Uses PyTorch's built-in GRU - more memory efficient and simpler than ConvGRU
    
    Args:
        num_classes: Number of classes (including background)
        hidden_channels: Hidden channels for plain GRU temporal processor
    
    Returns:
        FasterRCNN model with plain GRU temporal processing enabled
    """
    trainable_backbone_layers = 5
    trainable_backbone_layers = _validate_trainable_layers(True, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    # Create plain GRU temporal processor (most memory efficient)
    temporal_processor = PlainGRUTemporalProcessor(
        feature_channels=256,  # FPN feature channels
        hidden_channels=hidden_channels
    )

    return FasterRCNN(backbone, num_classes=num_classes, temporal_processor=temporal_processor)


def construct_temporal_model_conv_gru(num_classes=5, hidden_channels=128):
    """
    Construct a FasterRCNN model with ConvGRU temporal processing
    More memory efficient than LSTM version - recommended for memory-constrained environments
    
    Args:
        num_classes: Number of classes (including background)
        hidden_channels: Hidden channels for ConvGRU temporal processor
    
    Returns:
        FasterRCNN model with ConvGRU temporal processing enabled
    """
    trainable_backbone_layers = 5
    trainable_backbone_layers = _validate_trainable_layers(True, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, progress=True, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    # Create ConvGRU temporal processor (balances memory efficiency and spatial awareness)
    temporal_processor = GlobalPoolingTemporalProcessorGRU(
        feature_channels=256,  # FPN feature channels
        hidden_channels=hidden_channels
    )

    return FasterRCNN(backbone, num_classes=num_classes, temporal_processor=temporal_processor)

