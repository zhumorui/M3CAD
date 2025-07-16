# Eval Detection中Track Score过滤和AP计算分析

## 核心发现

通过分析代码，发现在eval detection时，**对boxes的过滤实际上包含了track_score的过滤**，而不仅仅是距离等信息的过滤。

## 详细分析

### 1. Track Score在推理中的作用

在`mmcv/core/bbox/coder/detr3d_track_coder.py`中的关键代码：

```python
# 第62行：使用track_scores进行top-k选择
_, bbox_index = track_scores.topk(max_num)

# 第70行：scores直接设置为track_scores
scores = track_scores

# 第73行：final_scores也是track_scores
final_scores = track_scores
```

### 2. 评估时的分数使用

在`mmcv/models/detectors/uniad_track.py`中：

```python
# 第1142行：scores_3d来自bboxes_dict["scores"]
scores_3d=scores.cpu(),

# 而scores来自decoder的输出，实际就是track_scores
```

在`mmcv/datasets/data_utils/data_utils.py`的`output_to_nusc_box`函数中：

```python
# 第18行：从detection结果中获取scores
scores = detection['scores_3d'].numpy()

# 第41行：box的score直接使用这个scores
score=scores[i],
```

### 3. 评估过程中的分数使用

在`mmcv/datasets/nuscenes_e2e_dataset.py`中：

```python
# 第821行：detection_score使用box.score
detection_score=box.score,
```

而在evaluation过程中（如`mmcv/datasets/eval_utils/eval_utils.py`）：

```python
# 第589行：使用detection_score作为置信度
pred_confs = [box.detection_score for box in pred_boxes_list]

# 第592行：按置信度排序进行匹配
sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
```

### 4. 过滤流程

评估过程中的过滤包括：

1. **Track Score过滤**：通过`track_scores.topk(max_num)`选择top boxes
2. **分数阈值过滤**：`final_scores > self.score_threshold`（其中final_scores就是track_scores）
3. **距离过滤**：在`filter_eval_boxes`中进行距离范围过滤
4. **NMS过滤**：如果启用NMS
5. **中心范围过滤**：`post_center_range`过滤

## 默认阈值设置

### Score Threshold默认值

在`mmcv/core/bbox/coder/detr3d_track_coder.py`中：
- **默认score_threshold = 0.2**

但在实际配置中（`mmcv/models/detectors/uniad_track.py`和`univ2x_track.py`）：
- **配置中的score_threshold = 0.0**

这意味着默认情况下：
- 代码中的默认值是0.2
- 但实际运行时配置设置为0.0，即**不进行track score阈值过滤**
- 只通过top-k选择（max_num=300）来限制boxes数量

### 其他相关阈值

1. **max_num = 300**：最多保留300个boxes
2. **score_thresh = 0.2**：在模型中的另一个分数阈值
3. **可视化阈值 = 0.25**：在可视化工具中使用（`m3cad/uniad/analysis_tools/visualize/run.py`）

## 结论

**代码在eval detection时，对boxes做了track_score的过滤再计算AP，而不是只过滤距离等信息。**

具体来说：
- `scores_3d`实际上就是`track_scores`
- 在生成评估用的detection boxes时，`detection_score`使用的是`track_scores`
- AP计算基于这些`detection_score`进行排序和匹配
- 因此track_score直接影响AP的计算结果
- **默认情况下score_threshold=0.0，主要通过top-k（max_num=300）进行过滤**

这意味着track_score不仅用于tracking，也直接影响detection的评估指标。