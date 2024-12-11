import numpy as np
import random
import math
from collections import defaultdict
from tqdm import tqdm
import os

def create_uniform_strategy(player_num):  # returns uniform strat and dict mapping infosets to tensor value
    if player_num == 0:
        lines = open(r'C:\Users\Alex\Downloads\pttt-infosets\player0-infoset.txt').readlines()
    else:
        lines = open(r'C:\Users\Alex\Downloads\pttt-infosets\player1-infoset.txt').readlines()
    tensor = np.ones((len(lines), 9), dtype=np.float64)
    infoset_index = dict()
    for idx, line in enumerate(lines):
        line = line.strip()  # Remove the \n
        infoset_index[line] = idx
        assert line.startswith('|')
        n = len(line) - 1
        assert n % 2 == 0
        n = n // 2
        for j in range(n):
            pos = line[1 + 2 * j]
            outcome = line[2 + 2 * j]
            assert outcome in '*.'
            assert pos in '012345678'
            pos = int(pos)
            tensor[idx, pos] = 0  # Set zero probability to illegal actions.
        if idx % 4000000 == 0:
            print(idx, 'done out of', len(lines))
    tensor /= np.sum(tensor, axis=1)[:, None]  # Renormalize row-wise
    return tensor, infoset_index


uniform_zero, _ = create_uniform_strategy(0)
uniform_one, _ = create_uniform_strategy(1)

base_path = r'C:\Users\Alex'  
counter = 0
for idx in tqdm(range(19800, -1, -200)):
    namezero = f'mccfr_pl0_{idx}.npy'
    nameone = f'mccfr_pl1_{idx}.npy'

    path_zero = os.path.join(base_path, namezero)
    path_one = os.path.join(base_path, nameone)

    arr_zero = np.load(path_zero)
    arr_one = np.load(path_one)

    for i in tqdm(range(len(arr_zero))):
        nonzero_mask = uniform_zero[i] > 0
        zero_mask = uniform_zero[i] == 0
        total_summation = np.sum(arr_zero[i])
        mask_summation = np.sum(arr_zero[i][nonzero_mask])
        if mask_summation <= 0:
            arr_zero[i] = uniform_zero[i]
        else:
            arr_zero[i][zero_mask] = 0
            arr_zero[i][nonzero_mask] *= 1/mask_summation
        if total_summation > mask_summation:
            counter += 1

    for i in tqdm(range(len(arr_one))):
        nonzero_mask = uniform_one[i] > 0
        zero_mask = uniform_one[i] == 0
        total_summation = np.sum(arr_one[i])
        mask_summation = np.sum(arr_one[i][nonzero_mask])
        if mask_summation <= 0:
            arr_one[i] = uniform_one[i]
        else:
            arr_one[i][zero_mask] = 0
            arr_one[i][nonzero_mask] *= 1/mask_summation
        if total_summation > mask_summation:
            counter += 1

    # Now arr_zero and arr_one are normalized row-wise.
    # If desired, you can save them back

    namezero = f'true_pl0_{idx}.npy'
    nameone = f'true_pl1_{idx}.npy'

    np.save(namezero, arr_zero)
    np.save(nameone, arr_one)

    print(counter)