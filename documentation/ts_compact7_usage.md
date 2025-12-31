# Using TS_Compact7_3x1x1 with nnUNetTrainerMRCT_AFP

## Overview
The `TS_Compact7_3x1x1` model is a compact TotalSegmentator-based feature extractor for the Anatomical Feature-Prioritized (AFP) loss. It uses a 7-class segmentation model trained on 3×1×1 mm resolution data with compact anatomical grouping.

## Class Mapping
The TotalSegmentator labels (100+ classes) are grouped into 6 anatomical categories plus background:

```python
CLASS_MAPPING = {
    "organs": 1,      # Liver, spleen, kidneys, etc.
    "cardiac": 2,     # Heart and major vessels
    "muscles": 3,     # Muscle tissues
    "bones": 4,       # General bone structures
    "ribs": 5,        # Rib cage
    "vertebrae": 6    # Vertebral column
}
```

## Configuration Details
- **Resolution**: 3×1×1 mm (adapted to dataset pixel spacing)
- **Architecture**: PlainConvUNet with 6 stages
- **Strides**: Optimized for anisotropic data `[[1,1,1], [1,2,2], [2,2,2], [2,2,2], [2,2,2], [1,2,2]]`
- **Kernels**: `[[1,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]`
- **Compatible datasets**: HN (Head & Neck), AB (Abdomen), TH (Thorax)

## Usage

### 1. Creating a Custom Trainer
To use the TS_Compact7_3x1x1 model, create a custom trainer by modifying the AFP initialization:

```python
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMRCT_AFP import nnUNetTrainerMRCT_AFP
from nnunetv2.training.loss.AFP import AFP
import torch

class nnUNetTrainerMRCT_AFP_TSCompact(nnUNetTrainerMRCT_AFP):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        # Initialize parent with custom AFP model
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Override the AFP loss with TS_Compact7_3x1x1 model
        self.AFP_loss = AFP(net="TS_Compact7_3x1x1", mae_weight=1.0)
```

### 2. Training
```bash
# Train with the new TS_Compact7_3x1x1 AFP model
nnUNetv2_train DatasetY 3d_fullres 0 \
  -tr nnUNetTrainerMRCT_AFP_TSCompact \
  -pl nnResUNetPlans
```

### 3. Alternative: Modify Existing Trainer
You can also modify the existing `nnUNetTrainerMRCT_AFP.py` to use the new model by changing line 36:

```python
# Before:
self.AFP_loss = AFP(net="TotalSeg_ABHNTH_117labels", mae_weight=1.0)

# After:
self.AFP_loss = AFP(net="TS_Compact7_3x1x1", mae_weight=1.0)
```

## Training the Segmentation Model
Before using TS_Compact7_3x1x1 for AFP loss, you need to train the segmentation model:

### 1. Prepare TotalSegmentator Labels
```bash
# Run TotalSegmentator on your CT volumes
totalsegmentator -i input_ct.nii.gz -o output_labels/
```

### 2. Group Labels into 7 Classes
Use the CLASS_MAPPING to convert TotalSegmentator's 100+ labels into 7 classes according to the anatomical groups.

### 3. Train the Compact Segmentation Model
```bash
# Prepare dataset with grouped labels
nnUNetv2_plan_and_preprocess -d DatasetX -c 3d_fullres

# Train the segmentation model
nnUNetv2_train DatasetX 3d_fullres 0
```

### 4. Update Weights Path
Update the weights path in `AFP.py` to point to your trained model:
```python
"TS_Compact7_3x1x1": {
    "weights_path": "/path/to/your/TS_Compact7_3x1x1/checkpoint_final.pth",
    ...
}
```

## Benefits
- **Reduced complexity**: Only 7 classes vs 100+ in standard TotalSegmentator
- **Dataset compatibility**: Works with HN, AB, and TH datasets
- **Resolution adapted**: Optimized for 3×1×1 mm spacing common in medical imaging
- **Easier training**: Simpler model is faster to train and requires less memory
- **Meaningful features**: Anatomical grouping provides semantically meaningful features for synthesis

## Reference
For more details, see the paper: [Cross-Anatomy CT Synthesis using Adapted nnResU-Net](https://arxiv.org/html/2509.22394)
