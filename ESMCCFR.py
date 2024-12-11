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

global regrets_zero, infoset_index_zero, regrets_one, infoset_index_one, strategy_sum_zero, strategy_sum_one, uniform_zero, uniform_one

uniform_zero, infoset_index_zero = create_uniform_strategy(0)
uniform_one, infoset_index_one = create_uniform_strategy(1)

regrets_zero = np.zeros(uniform_zero.shape)
regrets_one = np.zeros(uniform_one.shape)

strategy_sum_zero = np.zeros(uniform_zero.shape)
strategy_sum_one = np.zeros(uniform_one.shape)

def get_current_strategy(infoset, realization_weight, player):
    if player == 0:
        index = infoset_index_zero[infoset]
    else:
        index = infoset_index_one[infoset]
    positive_regrets = [max(r, 0.0) for r in regrets_zero[index]] if player == 0 else [max(r, 0.0) for r in regrets_one[index]]
    sum_positive = sum(positive_regrets)
    # If there are positive regrets, use them. Otherwise, use uniform strategy.
    if sum_positive > 0:
        strategy = [r / sum_positive for r in positive_regrets]
    else:
        if player == 0:
            strategy = uniform_zero[index] 
        else:
            strategy = uniform_one[index]
    strategy = np.array(strategy, dtype=np.float64)

    legal_actions = []

    if player == 0:
        index = infoset_index_zero[infoset]
        for i in range(9):
            if uniform_zero[index][i] > 0:
                legal_actions.append(i)
    else:
        index = infoset_index_one[infoset]
        for i in range(9):
            if uniform_one[index][i] > 0:
                legal_actions.append(i)

    for i in range(9):
        if i not in legal_actions:
            strategy[i] = 0
    
    # If realization_weight > 0, update strategy_sum for average strategy computation
    if realization_weight > 0:
        if player == 0:
            strategy_sum_zero[index] += strategy * realization_weight
        else:
            strategy_sum_one[index] += strategy * realization_weight

    return strategy

def observe_utility(infoset, cfs, player):
    if player == 0:
        index = infoset_index_zero[infoset]
    else:
        index = infoset_index_one[infoset]
    if player == 0:
        regrets_zero[index] += cfs
        #print("-------------UPDATE----------", infoset, player, cfs, regrets_zero[index])
    else:
        regrets_one[index] += cfs
        #print("-------------UPDATE----------",infoset, player, cfs, regrets_one[index])

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


def traverse_game(game, p0_infoset, p1_infoset, trained_player):
    #print("entering traverse game", game.board)
    if game.is_terminal():
        return game.get_reward()[trained_player]
    
    current_player = game.get_active_player()
    cp_infoset = p0_infoset if current_player == 0 else p1_infoset

    
    legal_actions = []

    if current_player == 0:
        index = infoset_index_zero[cp_infoset]
        for i in range(9):
            if uniform_zero[index][i] > 0:
                legal_actions.append(i)
    else:
        index = infoset_index_one[cp_infoset]
        for i in range(9):
            if uniform_one[index][i] > 0:
                legal_actions.append(i)

    #print("-------------ITERATION----------", current_player, cp_infoset, trained_player, legal_actions)
    if current_player != trained_player:
        action = sample_action(get_current_strategy(infoset = cp_infoset, realization_weight = 1.0, player = current_player))
        cp_infoset = cp_infoset + game.apply_move(action)
        game.next_player()
        if current_player == 0:
            reward = traverse_game(game, cp_infoset, p1_infoset, trained_player)
        else:
            reward = traverse_game(game, p0_infoset, cp_infoset, trained_player)
        if cp_infoset[-1] == "*":
            game.reverse_move(action)
        game.current_player = current_player
        return reward
    else:
        # Update for playerfff
        strategy = get_current_strategy(infoset = cp_infoset, realization_weight = 1.0, player = current_player)
        action_values = np.zeros(9)
        average_cf = 0.0
        for action in legal_actions:
            #print("legal action iter", action)
            next_state = cp_infoset + game.apply_move(action)
            game.next_player()
            if current_player == 0:
                reward = traverse_game(game, next_state, p1_infoset, trained_player)
            else:
                reward = traverse_game(game, p0_infoset, next_state, trained_player)
            action_values[action] = reward
            average_cf += reward*strategy[action]
            if next_state[-1] == "*":
                game.reverse_move(action)
            game.current_player = current_player
        
        for action in legal_actions:
            action_values[action] = action_values[action] - average_cf
        observe_utility(cp_infoset, action_values, current_player)

        return average_cf
    

   
def ExternalSamplingMCCFR(iterations, save_interval=1000):
    for idx in tqdm(range(iterations)):
        for player in [0,1]:
            game = Game()
            traverse_game(game, '|', '|', player)
            
        if  idx % save_interval == 0:
            # normalize strategy sum,make sure only assign positive values to those insid euniform , may take some time, need to implement
            namezero = 'MCCFR_pl0_' + str(idx) + '.npy'
            nameone = 'MCCFR_pl1_' + str(idx)+ '.npy'
            np.save(namezero, strategy_sum_zero)
            np.save(nameone, strategy_sum_one)
            print("downloaded ", idx)

ExternalSamplingMCCFR(20000, save_interval=200)