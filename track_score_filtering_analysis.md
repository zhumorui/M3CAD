# UniAD Track Score Filtering in Detection Evaluation Analysis

## Summary

**Yes, the UniAD codebase DOES apply track_score filtering to boxes during detection evaluation before calculating AP.** The filtering happens at multiple stages in the pipeline.

## Key Findings

### 1. Track Score-based Box Selection in DETR3D Coder

In `projects/mmdet3d_plugin/core/bbox/coders/detr3d_track_coder.py`, the `decode_single` method applies track_score filtering:

```python
# Line 62: Select top-k boxes based on track_scores
_, bbox_index = track_scores.topk(max_num)

# Lines 64-66: Filter all predictions using track_score ranking
labels = labels[bbox_index]
bbox_preds = bbox_preds[bbox_index]
track_scores = track_scores[bbox_index]
obj_idxes = obj_idxes[bbox_index]

# Line 69: Use track_scores as the final detection scores
scores = track_scores
final_scores = track_scores
```

### 2. Configuration Setting

The filtering is controlled by the `test_with_track_score=True` setting in the config file:
- Found in `projects/configs/stage2_e2e/base_e2e.py` line 396
- This setting enables track score filtering during inference

### 3. Detection Result Pipeline

The detection evaluation follows this pipeline:

1. **Model Forward Pass**: Generates track_scores for each query
2. **DETR3D Track Coder**: Applies `track_scores.topk(max_num)` to select boxes
3. **Result Formatting**: Uses filtered boxes for detection evaluation
4. **AP Calculation**: Standard NuScenes evaluation on the filtered box set

### 4. Key Code Locations

- **Track Score Generation**: `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` lines 411, 676
- **Track Score Filtering**: `projects/mmdet3d_plugin/core/bbox/coders/detr3d_track_coder.py` lines 62-69
- **Detection Result Creation**: `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` lines 825-831
- **Result Formatting**: `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py` lines 873-947

### 5. Impact on AP Calculation

The track_score filtering affects AP calculation in two ways:

1. **Box Selection**: Only the top-k boxes ranked by track_score are considered for evaluation
2. **Score Assignment**: The track_scores become the detection scores used in AP calculation

This means that boxes with low track_scores are filtered out before the standard NuScenes detection evaluation pipeline, potentially improving AP metrics by removing low-confidence tracked objects.

### 6. Occupancy Head Integration

The `test_with_track_score` setting also affects occupancy prediction:
- In `projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py` lines 436-439
- Track scores are multiplied with occupancy predictions when enabled

## Conclusion

UniAD explicitly uses track_scores to filter detection boxes during evaluation. This is a key design choice that leverages the tracking component's confidence scores to improve detection performance metrics. The filtering occurs before the standard NuScenes AP calculation, making track_score an integral part of the detection evaluation process.