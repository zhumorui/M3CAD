# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

import json
import os
from typing import Dict, List

from nuscenes import NuScenes





# M3CAD dataset
train = [
    # Timestamp: 2025_06_24_18_33_22
    '2025_06_24_18_33_22_60', '2025_06_24_18_33_22_51', '2025_06_24_18_33_22_75',
    
    # Timestamp: 2025_03_03_15_02_55
    '2025_03_03_15_02_55_51', '2025_03_03_15_02_55_50', '2025_03_03_15_02_55_37',
    
    # Timestamp: 2025_07_01_18_22_24
    '2025_07_01_18_22_24_75', '2025_07_01_18_22_24_60', '2025_07_01_18_22_24_51',
    
    # Timestamp: 2025_03_03_19_46_59
    '2025_03_03_19_46_59_37', '2025_03_03_19_46_59_47', '2025_03_03_19_46_59_25',
    
    # Timestamp: 2025_03_03_17_27_46
    '2025_03_03_17_27_46_25', '2025_03_03_17_27_46_37',
    
    # Timestamp: 2025_03_03_17_43_14
    '2025_03_03_17_43_14_25', '2025_03_03_17_43_14_37', '2025_03_03_17_43_14_47',
    
    # Timestamp: 2025_03_03_20_26_07
    '2025_03_03_20_26_07_47', '2025_03_03_20_26_07_51', '2025_03_03_20_26_07_37', '2025_03_03_20_26_07_50',
    
    # Timestamp: 2025_06_07_23_42_19
    '2025_06_07_23_42_19_85', '2025_06_07_23_42_19_75', '2025_06_07_23_42_19_81',
    
    # Timestamp: 2025_03_03_16_08_34
    '2025_03_03_16_08_34_47', '2025_03_03_16_08_34_37', '2025_03_03_16_08_34_50', '2025_03_03_16_08_34_25',
    
    # Timestamp: 2025_05_27_14_08_01
    '2025_05_27_14_08_01_60', '2025_05_27_14_08_01_50', '2025_05_27_14_08_01_51',
    
    # Timestamp: 2025_03_03_15_35_50
    '2025_03_03_15_35_50_37', '2025_03_03_15_35_50_25', '2025_03_03_15_35_50_47',
    
    # Timestamp: 2025_03_03_19_35_10
    '2025_03_03_19_35_10_47', '2025_03_03_19_35_10_25', '2025_03_03_19_35_10_50', '2025_03_03_19_35_10_37',
    
    # Timestamp: 2025_06_22_18_33_22
    '2025_06_22_18_33_22_51', '2025_06_22_18_33_22_60', '2025_06_22_18_33_22_75',
    
    # Timestamp: 2025_03_03_18_13_10
    '2025_03_03_18_13_10_37', '2025_03_03_18_13_10_50', '2025_03_03_18_13_10_47', '2025_03_03_18_13_10_51',
    
    # Timestamp: 2025_03_03_16_22_16
    '2025_03_03_16_22_16_47', '2025_03_03_16_22_16_25', '2025_03_03_16_22_16_37',
    
    # Timestamp: 2025_03_03_15_17_51
    '2025_03_03_15_17_51_51', '2025_03_03_15_17_51_47', '2025_03_03_15_17_51_50', '2025_03_03_15_17_51_37',
    
    # Timestamp: 2025_07_11_09_04_49
    '2025_07_11_09_04_49_75', '2025_07_11_09_04_49_60', '2025_07_11_09_04_49_51',
    
    # Timestamp: 2025_03_03_20_19_40
    '2025_03_03_20_19_40_37', '2025_03_03_20_19_40_25', '2025_03_03_20_19_40_47',
    
    # Timestamp: 2025_03_03_17_32_03
    '2025_03_03_17_32_03_37', '2025_03_03_17_32_03_47', '2025_03_03_17_32_03_25',
    
    # Timestamp: 2025_06_29_12_50_21
    '2025_06_29_12_50_21_50', '2025_06_29_12_50_21_60', '2025_06_29_12_50_21_51',
    
    # Timestamp: 2025_03_03_19_02_35
    '2025_03_03_19_02_35_37', '2025_03_03_19_02_35_50', '2025_03_03_19_02_35_47', '2025_03_03_19_02_35_25',
    
    # Timestamp: 2025_03_03_14_52_14
    '2025_03_03_14_52_14_37', '2025_03_03_14_52_14_51', '2025_03_03_14_52_14_47',
    
    # Timestamp: 2025_05_26_14_08_01
    '2025_05_26_14_08_01_51', '2025_05_26_14_08_01_60', '2025_05_26_14_08_01_50',
    
    # Timestamp: 2025_03_03_16_56_12
    '2025_03_03_16_56_12_37', '2025_03_03_16_56_12_50', '2025_03_03_16_56_12_47',
    
    # Timestamp: 2025_03_03_18_08_10
    '2025_03_03_18_08_10_25', '2025_03_03_18_08_10_47', '2025_03_03_18_08_10_37', '2025_03_03_18_08_10_50',
    
    # Timestamp: 2025_06_12_10_31_09
    '2025_06_12_10_31_09_60', '2025_06_12_10_31_09_75', '2025_06_12_10_31_09_51',
    
    # Timestamp: 2025_03_03_15_05_32
    '2025_03_03_15_05_32_37', '2025_03_03_15_05_32_51', '2025_03_03_15_05_32_47', '2025_03_03_15_05_32_50',
    
    # Timestamp: 2025_05_31_09_26_05
    '2025_05_31_09_26_05_51', '2025_05_31_09_26_05_75', '2025_05_31_09_26_05_60',
    
    # Timestamp: 2025_03_03_16_06_25
    '2025_03_03_16_06_25_47', '2025_03_03_16_06_25_37', '2025_03_03_16_06_25_25',
    
    # Timestamp: 2025_03_03_17_05_14
    '2025_03_03_17_05_14_25', '2025_03_03_17_05_14_47',
    
    # Timestamp: 2025_03_03_20_40_30
    '2025_03_03_20_40_30_50', '2025_03_03_20_40_30_51', '2025_03_03_20_40_30_47', '2025_03_03_20_40_30_37',
    
    # Timestamp: 2025_03_03_19_26_56
    '2025_03_03_19_26_56_25', '2025_03_03_19_26_56_47', '2025_03_03_19_26_56_37',
    
    # Timestamp: 2025_03_03_15_11_06
    '2025_03_03_15_11_06_50', '2025_03_03_15_11_06_47', '2025_03_03_15_11_06_51', '2025_03_03_15_11_06_37',
    
    # Timestamp: 2025_07_07_08_53_39
    '2025_07_07_08_53_39_75', '2025_07_07_08_53_39_60', '2025_07_07_08_53_39_51',
    
    # Timestamp: 2025_03_03_17_18_27
    '2025_03_03_17_18_27_50', '2025_03_03_17_18_27_37', '2025_03_03_17_18_27_47', '2025_03_03_17_18_27_51',
    
    # Timestamp: 2025_06_04_18_44_22
    '2025_06_04_18_44_22_60', '2025_06_04_18_44_22_50', '2025_06_04_18_44_22_51',
    
    # Timestamp: 2025_03_03_19_42_20
    '2025_03_03_19_42_20_50', '2025_03_03_19_42_20_37', '2025_03_03_19_42_20_47', '2025_03_03_19_42_20_25',
    
    # Timestamp: 2025_03_03_16_15_53
    '2025_03_03_16_15_53_37', '2025_03_03_16_15_53_47', '2025_03_03_16_15_53_25',
    
    # Timestamp: 2025_05_30_09_26_05
    '2025_05_30_09_26_05_60', '2025_05_30_09_26_05_51', '2025_05_30_09_26_05_75',
    
    # Timestamp: 2025_03_03_20_00_30
    '2025_03_03_20_00_30_47', '2025_03_03_20_00_30_25',
    
    # Timestamp: 2025_03_03_16_36_16
    '2025_03_03_16_36_16_47', '2025_03_03_16_36_16_50', '2025_03_03_16_36_16_51',
    
    # Timestamp: 2025_03_03_17_23_10
    '2025_03_03_17_23_10_47', '2025_03_03_17_23_10_37', '2025_03_03_17_23_10_25',
    
    # Timestamp: 2025_06_02_18_44_22
    '2025_06_02_18_44_22_60', '2025_06_02_18_44_22_50', '2025_06_02_18_44_22_51',
    
    # Timestamp: 2025_03_03_19_09_29
    '2025_03_03_19_09_29_37', '2025_03_03_19_09_29_25', '2025_03_03_19_09_29_47',
    
    # Timestamp: 2025_07_05_14_39_47
    '2025_07_05_14_39_47_60', '2025_07_05_14_39_47_51', '2025_07_05_14_39_47_50',
]

val = [
   # Timestamp: 2025_03_03_15_30_41
    '2025_03_03_15_30_41_25', '2025_03_03_15_30_41_50', '2025_03_03_15_30_41_47',
    
    # Timestamp: 2025_03_03_21_08_04
    '2025_03_03_21_08_04_51', '2025_03_03_21_08_04_50', '2025_03_03_21_08_04_37',
    
    # Timestamp: 2025_03_03_18_58_40
    '2025_03_03_18_58_40_37', '2025_03_03_18_58_40_47', '2025_03_03_18_58_40_25',
    
    # Timestamp: 2025_03_03_19_05_10
    '2025_03_03_19_05_10_47', '2025_03_03_19_05_10_37', '2025_03_03_19_05_10_25',
    
    # Timestamp: 2025_03_03_16_00_01
    '2025_03_03_16_00_01_37', '2025_03_03_16_00_01_25', '2025_03_03_16_00_01_47',
    
    # Timestamp: 2025_03_03_19_49_04
    '2025_03_03_19_49_04_50', '2025_03_03_19_49_04_25', '2025_03_03_19_49_04_37', '2025_03_03_19_49_04_47',
    
    # Timestamp: 2025_06_03_18_44_22
    '2025_06_03_18_44_22_60', '2025_06_03_18_44_22_50', '2025_06_03_18_44_22_51',
    
    # Timestamp: 2025_05_24_14_08_01
    '2025_05_24_14_08_01_60', '2025_05_24_14_08_01_50', '2025_05_24_14_08_01_51',
    
    # Timestamp: 2025_03_03_15_28_34
    '2025_03_03_15_28_34_25', '2025_03_03_15_28_34_47', '2025_03_03_15_28_34_37',
    
    # Timestamp: 2025_03_03_20_21_46
    '2025_03_03_20_21_46_37', '2025_03_03_20_21_46_47',
]

test = [
    # Timestamp: 2025_03_03_14_54_49
    '2025_03_03_14_54_49_47', '2025_03_03_14_54_49_37', '2025_03_03_14_54_49_50', '2025_03_03_14_54_49_51',

    # Timestamp: 2025_03_03_17_45_21
    '2025_03_03_17_45_21_25', '2025_03_03_17_45_21_37', '2025_03_03_17_45_21_47', '2025_03_03_17_45_21_50',
    
    # Timestamp: 2025_06_11_10_31_09
    '2025_06_11_10_31_09_51', '2025_06_11_10_31_09_60', '2025_06_11_10_31_09_75',
    
    # Timestamp: 2025_07_02_06_03_04
    '2025_07_02_06_03_04_85', '2025_07_02_06_03_04_75', '2025_07_02_06_03_04_81',
    
    # Timestamp: 2025_03_03_17_07_25
    '2025_03_03_17_07_25_25', '2025_03_03_17_07_25_47', '2025_03_03_17_07_25_50', '2025_03_03_17_07_25_37',
    
    # Timestamp: 2025_05_29_10_09_36
    '2025_05_29_10_09_36_60', '2025_05_29_10_09_36_75', '2025_05_29_10_09_36_51',
    
    # Timestamp: 2025_03_03_16_11_05
    '2025_03_03_16_11_05_47', '2025_03_03_16_11_05_50', '2025_03_03_16_11_05_37', '2025_03_03_16_11_05_51',
    
    # Timestamp: 2025_06_23_18_33_22
    '2025_06_23_18_33_22_51', '2025_06_23_18_33_22_75', '2025_06_23_18_33_22_60',
    
    # Timestamp: 2025_03_03_18_43_56
    '2025_03_03_18_43_56_47', '2025_03_03_18_43_56_25',
]


train_detect = train
train_track = train

# FIXME
mini_train = train
mini_val = val

def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


def get_scenes_of_split(split_name: str, nusc : NuScenes, verbose: bool = False) -> List[str]:
    """
    Returns the scenes in a given split.
    :param split_name: The name of the split.
    :param nusc: The NuScenes instance to know where to look up potential custom splits.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A list of scenes in that split.
    """

    if is_predefined_split(split_name=split_name):
        return create_splits_scenes(verbose=verbose)[split_name]
    else:
        return get_scenes_of_custom_split(split_name=split_name, nusc=nusc)

def is_predefined_split(split_name: str) -> bool:
    """
    Returns whether the split name is one of the predefined splits in the nuScenes dataset.
    :param split_name: The name of the split.
    :return: Whether the split is predefined.
    """
    return split_name in create_splits_scenes().keys()


def get_scenes_of_custom_split(split_name: str, nusc : NuScenes) -> List[str]:
    """Returns the scene names from a custom `splits.json` file."""

    splits_file_path: str = _get_custom_splits_file_path(nusc)

    splits_data: dict = {}
    with open(splits_file_path, 'r') as file:
        splits_data = json.load(file)

    if split_name not in splits_data.keys():
        raise ValueError(f"Custom split {split_name} requested, but not found in {splits_file_path}.")

    scene_names_of_split : List[str] = splits_data[split_name]
    assert isinstance(scene_names_of_split, list), \
        f'Custom split {split_name} must be a list of scene names in {splits_file_path}.'
    return scene_names_of_split


def _get_custom_splits_file_path(nusc : NuScenes) -> str:
    """Use a separate function for this so we can mock it well in unit tests."""

    splits_file_path: str = os.path.join(nusc.dataroot, nusc.version, "splits.json")
    if (not os.path.exists(splits_file_path)) or (not os.path.isfile(splits_file_path)):
        raise ValueError(f"Custom split requested, but no valid file found at {splits_file_path}.")

    return splits_file_path


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
