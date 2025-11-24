# Fauna Training - Blocked on num_iters Issue

**Status**: BLOCKED - num_iters config not being read correctly

## Problem

Training runs for only 1 iteration instead of 50,000 despite `cfg_trainer.num_iters: 50000` being set in config.

**Log Output**:
```
T000001/    1.5Hz	loss: 34.87223	...
Training completed for all 1 iterations.
```

## Attempted Fixes

1. ✅ Changed section name: `training:` → `cfg_trainer:`
2. ❌ Still defaulting to 1 iteration (TrainerConfig default)

## Root Cause (Suspected)

Hydra config composition/override order issue. The `cfg_trainer` section in `train_fauna_mouse_finetune.yaml` may not be merging correctly with base config.

## Next Steps (For Tomorrow)

1. **Check base_fauna.yaml**: See if it defines cfg_trainer with num_iters
2. **Test with override**: Try `python run.py --config-name train_fauna_mouse_finetune cfg_trainer.num_iters=50000`
3. **Alternative**: Create standalone config without inheritance
4. **Debug print**: Add print statement in Trainer.py `__init__` to see actual num_iters value

## Workaround (Last Resort)

Temporarily hard-code in `/home/joon/dev/3DAnimals/model/Trainer.py:24`:
```python
num_iters: int = 50000  # TEMPORARY FOR FAUNA MOUSE
```

## All Other Issues RESOLVED ✅

- ✅ Python dataclass `pretrained_sdf` field
- ✅ Absolute paths
- ✅ Empty directories created
- ✅ Metadata fields (video_frame_id, crop_box_xyxy, etc.)
- ✅ `load_dino_feature: false`
- ✅ Images resized to 256×256
- ✅ GPU memory available (11.7GB free)
- ✅ Config section name (`cfg_trainer` not `training`)

## Current State

- **Data**: Ready (50 frames, 256×256, all metadata correct)
- **Config**: 99% ready (only num_iters issue)
- **GPU**: Ready (fully available)
- **Code**: All fixes applied
- **Blocking**: num_iters config not loading

**Estimated time to fix**: 10-30 minutes once root cause identified
