import numpy as np
import random
import math
from collections import defaultdict
from tqdm import tqdm

import os
print("Current Working Directory:", os.getcwd())

def create_uniform_strategy(player_num):  # returns uniform strat and dict mapping infosets to tensor value
    if player_num == 0:
        lines = open(r'C:\Users\Alex\Downloads\pttt-infosets\player0-infoset.txt').readlines()
    else:
        lines = open(r'C:\Users\Alex\Downloads\pttt-infosets\player1-infoset.txt').readlines()
    tensor = np.ones((len(lines), 9), dtype=np.float64)
    infoset_index = dict()
    for idx, line in tqdm(enumerate(lines)):
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
    tensor /= np.sum(tensor, axis=1)[:, None]  # Renormalize row-wise
    return tensor, infoset_index

global infoset_index_zero, infoset_index_one, uniform_zero, uniform_one

uniform_one, infoset_index_one = create_uniform_strategy(1)
_, infoset_index_zero = create_uniform_strategy(0)

class Game:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 0
        self.win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]

    def get_active_player(self):
        return self.current_player
    
    def copy(self):
        new_game = Game()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def next_player(self):
        self.current_player = 1 if self.current_player == 0 else 0

    def apply_move(self, move):
        if self.board[move] == ' ':
            self.board[move] = 'X' if self.current_player == 0 else 'O'
            return f'{move}*'
        else:
            return f'{move}.'
        
    def reverse_move(self, move):
        self.board[move] = ' '

    def is_terminal(self):
        counter = 0
        for pos in self.win_positions:
            if self.board[pos[0]] == self.board[pos[1]] == self.board[pos[2]] != ' ':
                return True
            line = {self.board[p] for p in pos}
            if 'X' in line and 'O' in line:
                counter += 1
        if counter == 8:
            return True
        if ' ' not in self.board:
            return True
        return False

    def get_reward(self):
        for pos in self.win_positions:
            if self.board[pos[0]] == self.board[pos[1]] == self.board[pos[2]] != ' ':
                if self.board[pos[0]] == 'X':
                    return (1, -1)  # Player 0 wins
                else:
                    return (-1, 1)  # Player 1 wins
        if ' ' not in self.board:
            return (0, 0)  # Tie
        return (0, 0)  # Game not finished
    
def sample_action(strategy):
    """
    Given a probability distribution over actions, return an action index.
    """
    return np.random.choice(9, p=strategy)


def iter_game(game, p0_infoset, p1_infoset, current_player):
    #print("entering traverse game", game.board)
    if game.is_terminal():
        reward = game.get_reward()
        if reward [0] == 1:
            return "win"
        elif reward[0] == 0:
            return "tie"
        else:
            return "lose"
    
    if current_player != game.get_active_player():
        exit()

    if current_player == 0:
        action = sample_action(arr_zero[infoset_index_zero[p0_infoset]])
        p0_infoset = p0_infoset + game.apply_move(action)
    else:
        action = sample_action(uniform_one[infoset_index_one[p1_infoset]])
        p1_infoset = p1_infoset + game.apply_move(action)
    
    game.next_player()
    return iter_game(game, p0_infoset, p1_infoset, 1-current_player)
    

   
def winrate(iterations):
    win_counter = 0.0
    tie_counter = 0.0
    lose_counter = 0.0
    for idx in range(iterations):
        game = Game()
        winner = iter_game(game, '|', '|', 0)
        if winner == "win":
            win_counter += 1
        elif winner == "tie":
            tie_counter += 1
        else:
            lose_counter += 1
    total = win_counter + tie_counter + lose_counter 
    return (win_counter/total, tie_counter/total, lose_counter/total)

total = []

base_path = r'C:\Users\Alex'  
for idx in tqdm(range(19800, -1, -200)):
    namezero = f'true_pl0_{idx}.npy'

    path_zero = os.path.join(base_path, namezero)

    arr_zero = np.load(path_zero)

    total.append([idx, winrate(100000)])
print(total)
total = np.array(total, dtype=np.float64)
np.save("winrates.npy", total)