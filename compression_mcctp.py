import numpy as np
import random
import math
from collections import defaultdict
from tqdm import tqdm
import os


base_path = r'C:\Users\Alex'  
counter = 0
for idx in tqdm(range(120000000-1200000, -1, -1200000)):
    namezero = f'mcts_pttt_pl0_{idx}.npy'
    nameone = f'mcts_pttt_pl1_{idx}.npy'

    path_zero = os.path.join(base_path, namezero)
    path_one = os.path.join(base_path, nameone)

    arr_zero = np.load(path_zero)
    arr_one = np.load(path_one)

    arr_zero = arr_zero.astype(np.float32)
    arr_one = arr_one.astype(np.float32)

    threshold = 1e-3  # Adjust as needed

    for i in range(len(arr_zero)):
        for j in range(9):
            if arr_zero[i][j] < threshold:
                arr_zero[i][j] = 0
        arr_zero[i] = arr_zero[i]/np.sum(arr_zero[i]) 

    for i in range(len(arr_one)):
        for j in range(9):
            if arr_one[i][j] < threshold:
                arr_one[i][j] = 0
        arr_one[i] = arr_one[i]/np.sum(arr_one[i]) 

    namezero = f'mctsthresh_pl0_{idx}.npy'
    nameone = f'mctsthresh_pl1_{idx}.npy'

    np.save(namezero, arr_zero)
    np.save(nameone, arr_one)