# Quick Start: Using TS_Compact7_3x1x1 with AFP Loss

There are **two ways** to use the new TS_Compact7_3x1x1 model:

## Option 1: Use the New Dedicated Trainer (Recommended)

The easiest way is to use the pre-configured trainer variant:

```bash
nnUNetv2_train DatasetY 3d_fullres 0 \
  -tr nnUNetTrainerMRCT_AFP_TSCompact \
  -pl nnResUNetPlans
```

This trainer automatically uses the TS_Compact7_3x1x1 model for AFP loss.

## Option 2: Modify the Existing nnUNetTrainerMRCT_AFP

If you prefer to use the existing `nnUNetTrainerMRCT_AFP` trainer, simply change one line:

**File:** `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_AFP.py`

**Line 36:** Change from:
```python
self.AFP_loss = AFP(net="TotalSeg_ABHNTH_117labels", mae_weight=1.0)
```

**To:**
```python
self.AFP_loss = AFP(net="TS_Compact7_3x1x1", mae_weight=1.0)
```

Then use it as before:
```bash
nnUNetv2_train DatasetY 3d_fullres 0 \
  -tr nnUNetTrainerMRCT_AFP \
  -pl nnResUNetPlans
```

## Available Models in AFP

You can now choose from these TotalSegmentator-based models:

| Model Name | Resolution | Classes | Description |
|------------|------------|---------|-------------|
| `TotalSeg_ABHNTH_117labels` | 3×1×1 mm | 118 | Full TotalSegmentator labels |
| `TotalSeg_ABHNTH_20labels` | 3×1×1 mm | 21 | 20 grouped labels |
| `TS_Compact7_3x1x1` ⭐ | 3×1×1 mm | 7 | **NEW** Compact 6-class anatomical grouping |
| `TotalSeg_AB_7labels` | 3×1×1 mm | 7 | Abdomen-specific 7 labels |
| `TotalSeg117` | 1.5×1.5×1.5 mm | 118 | High-resolution full labels |
| `TotalSeg_V2` | 1.5×1.5×1.5 mm | 8 | High-resolution 7 labels |

## Notes

- ⚠️ Before using any model, you must **train the segmentation network** first (see `documentation/ts_compact7_usage.md` for details)
- The weights path in `AFP.py` points to the expected location - update it to match your trained model path
- The TS_Compact7_3x1x1 model is specifically designed for 3×1×1 mm spacing and works with HN, AB, and TH datasets

## Next Steps

For detailed information about training the segmentation model and understanding the class mapping, see:
- `documentation/ts_compact7_usage.md` - Complete usage guide
- [arXiv:2509.22394](https://arxiv.org/html/2509.22394) - Original paper
