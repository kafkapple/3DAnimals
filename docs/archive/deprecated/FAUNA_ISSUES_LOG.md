# Fauna Training Issues & Solutions

## Issue #6: CUDA Out of Memory (Image Size)
**Date**: 2025-11-11 21:44

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 30.38 GiB
(GPU 0; 11.75 GiB total capacity; 1.09 GiB already allocated; 6.67 GiB free)
```

**Cause**:
- DANNCE data: 1152×1024 images
- Config: `in_image_size: 1024`
- ViT processes 1024×1024 patches → 30.38 GiB memory needed
- RTX 3060: Only 11.75 GiB total

**Root Problem**:
Data conversion script saved original resolution (1152×1024) instead of resizing to 256×256 as Fauna expects.

**Solution** (Pending):
1. **Option A** (Recommended): Re-run data conversion with resize
   ```bash
   cd /home/joon/dev/project_splatter
   python scripts/convert_dannce_to_fauna.py \
     --dannce_root data/dannce_mouse_6view \
     --output_root data/fauna_mouse \
     --extract_views best \
     --resize 256 \
     --num_workers 4
   ```

2. **Option B**: Modify existing images with PIL
   - Resize all 150 RGB + mask files
   - Update metadata.json with new dimensions

**Config Changes Needed**:
```yaml
# /home/joon/dev/3DAnimals/config/train_fauna_mouse_finetune.yaml
dataset:
  in_image_size: 256    # Match resized data
  out_image_size: 256   # Training resolution
```

**Status**: Blocked - Need to resize data before training can proceed

---

## Previous Issues (Resolved)

### Issue #5: Tensor Size Mismatch
- **Error**: `RuntimeError: The size of tensor a (256) must match the size of tensor b (288)`
- **Cause**: Metadata `crop_box_xyxy` was [0,0,256,256] but actual images were 1152×1024
- **Solution**: Updated metadata with correct dimensions ✅

### Issue #4: DINO Feature File Missing
- **Error**: `FileNotFoundError: '0005165_feat16.png'`
- **Solution**: Set `load_dino_feature: false` in config ✅

### Issue #3: Metadata Field Missing (crop_box_xyxy)
- **Error**: `TypeError: cannot unpack non-iterable NoneType object`
- **Solution**: Added `crop_box_xyxy`, `video_frame_width/height` fields ✅

### Issue #2: Metadata Field Missing (video_frame_id)
- **Error**: `TypeError: int() argument must be a string... not 'NoneType'`
- **Solution**: Created metadata.json with `video_frame_id` field ✅

### Issue #1: Python Dataclass Field Missing
- **Error**: `AttributeError: 'DMTetEmbConfig' object has no attribute 'pretrained_sdf'`
- **Solution**: Added field to BasePredictorBank.py dataclass ✅

---

## Next Actions

1. **Immediate**: Add `--resize 256` to data conversion script
2. **Verify**: Check if convert_dannce_to_fauna.py supports resize parameter
3. **Execute**: Re-run conversion (will overwrite existing data)
4. **Update**: Config back to `in_image_size: 256`
5. **Retry**: Launch training

**Estimated Time**: 10-15 minutes (data conversion + training start)
