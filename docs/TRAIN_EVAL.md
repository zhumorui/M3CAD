# Training and Evaluation

## Training Pipeline
### Non-Cooperative Training

Stage1: Tracking and Mapping

```bash
./m3cad/uniad/uniad_dist_train.sh m3cad/uniad/configs/stage1_track_map/base_track_map.py 4
```

Stage2: End-to-End

```bash
./m3cad/uniad/uniad_dist_train.sh m3cad/uniad/configs/stage2_e2e/base_e2e.py 4
```

### Cooperative Training

Stage1: Tracking and Mapping
```bash
./m3cad/uniad/uniad_dist_train.sh m3cad/uniad/configs/stage1_cooper_track_map/base_track_map.py 4
```

Stage2: End-to-End
```bash
./m3cad/uniad/uniad_dist_train.sh m3cad/uniad/configs/stage2_cooper_e2e/base_e2e.py 4
```


## Evaluation

### Non-Cooperative Evaluation
```bash
./m3cad/uniad/uniad_dist_test.sh m3cad/uniad/configs/stage2_e2e/base_e2e.py 4
```

### Cooperative Evaluation
```bash
./m3cad/uniad/uniad_dist_test.sh m3cad/uniad/configs/stage2_cooper_e2e/base_e2e.py 4
```
