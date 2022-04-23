import os
import numpy as np
from copy import deepcopy
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image
import torch

def generate_trigger(trigger_type, s=28):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(s, s, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(s, s, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[s-2 + h, s-2 + w, 0] = trigger_value[h+1][w+1]
                mask[s-2 + h, s-2 + w, 0] = 1
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(s, s, 1), dtype=np.uint8)
        mask = np.zeros(shape=(s, s, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, s-2]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, s-2 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, s-2 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
    elif trigger_type == 'gaussian_noise':
        pattern = np.array(Image.open('./cifar_gaussian_noise.png'))
        mask = np.ones(shape=(s, s, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def add_trigger(data_set, trigger_type, poison_rate, poison_target, trigger_alpha, s=32):
    """
    A simple implementation for backdoor attacks which only supports Badnets and Blend.
    :param clean_set: The original clean data.
    :param poison_type: Please choose on from [checkerboard_1corner | checkerboard_4corner | gaussian_noise].
    :param poison_rate: The injection rate of backdoor attacks.
    :param poison_target: The target label for backdoor attacks.
    :param trigger_alpha: The transparency of the backdoor trigger.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """
    pattern, mask = generate_trigger(trigger_type=trigger_type, s=s)
    poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
    poison_set = deepcopy(data_set)
    poison_num = int(poison_rate * len(poison_cand))
    choices = np.random.choice(poison_cand, poison_num, replace=False)

    for idx in choices:
        orig = poison_set.data[idx]
        poison_set.data[idx] = np.clip(
            (1 - mask) * orig + mask * ((1 - trigger_alpha) * orig + trigger_alpha * pattern), 0, 255
        ).astype(np.uint8)
        poison_set.targets[idx] = poison_target
    trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                    'trigger_alpha': trigger_alpha, 'poison_target': np.array([poison_target]),
                    'data_index': choices}
    return poison_set, trigger_info

import scipy.ndimage as nd
def add_predefined_trigger(data_set, trigger_info, exclude_target=True):
    """
    Poisoning dataset using a predefined trigger.
    This can be easily extended to various attacks as long as they provide trigger information for every sample.
    :param data_set: The original clean dataset.
    :param trigger_info: The information for predefined trigger.
    :param exclude_target: Whether to exclude samples that belongs to the target label.
    :return: A poisoned dataset
    """
    if trigger_info is None:
        return data_set
    poison_set = deepcopy(data_set)

    pattern = trigger_info['trigger_pattern']
    mask = trigger_info['trigger_mask']
    alpha = trigger_info['trigger_alpha']
    poison_target = trigger_info['poison_target']
    poison_set.data = \
        ((1 - mask) * poison_set.data + mask * ((1 - alpha) * poison_set.data + alpha * pattern)).astype(np.uint8)
    if poison_target.size == 1:
        poison_target = np.repeat(poison_target, len(poison_set.targets), axis=0)
        poison_target = np.array(poison_target)
    poison_set.targets = poison_target

    if exclude_target:
        no_target_idx = (poison_target != data_set.targets)
        poison_set.data = poison_set.data[no_target_idx, :, :, :]
        poison_set.targets = poison_set.targets[no_target_idx]
    return poison_set

