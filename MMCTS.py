import numpy as np
from tqdm import tqdm
import random
import math
from collections import defaultdict

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
global reward_zero, infoset_index_zero
global reward_one, infoset_index_one

reward_zero, infoset_index_zero = create_uniform_strategy(0)
reward_one, infoset_index_one = create_uniform_strategy(1)

uniform_zero = np.array(reward_zero)
uniform_one = np.array(reward_one)

class Node:
    def __init__(self, player, infoset = "|", parent=None, move=None, depth=0, probability = float(1/9), in_tree = False):
        self.player = player
        self.parent = parent
        self.children = dict()
        self.move = move
        self.probability = probability
        self.depth = depth
        self.infoset = infoset
        self.in_tree = in_tree
        #self.regret_sum = defaultdict(float)  # For MCCFR
        #self.strategy_sum = defaultdict(float)  # Cumulative strategy for MCCFR

    def sample_action(self, algo, gamma):
        if algo == "mmcts":
            if self.player == 0:
                for i in range(9):
                    if uniform_zero[infoset_index_zero[self.infoset]][i] == 0:
                        reward_zero[infoset_index_zero[self.infoset]][i] = 0
                rewards = reward_zero[infoset_index_zero[self.infoset]]
            else:
                for i in range(9):
                    if uniform_one[infoset_index_one[self.infoset]][i] == 0:
                        reward_one[infoset_index_one[self.infoset]][i] = 0
                rewards = reward_one[infoset_index_one[self.infoset]]
            #print("sampling action for player ", self.player," at ",self.infoset, " with rewards ", rewards, " intree = ",self.in_tree)
            if self.in_tree:
                # MMCTS strategy based on EXP3 probabilities
                k = np.count_nonzero(rewards) # each non zero entry is a valid move
                probabilities = rewards / np.sum(rewards)           
                probabilities = probabilities/np.sum(probabilities)
                #print("probabilities: ",probabilities, " gamma ", gamma, " k ", k)
                self.move = np.random.choice(len(rewards), p=probabilities)
                #print("move ", self.move)
                return (self.move, probabilities[self.move])
            else:
                probabilities = rewards/np.sum(rewards)
                #print("probabilities: ",probabilities," gamma ", gamma)
                self.move = np.random.choice(len(rewards), p=probabilities)
                #print("move ",self.move)
                return (self.move, probabilities[self.move])
        
        elif algo == "mccfr":
            # MCCFR strategy based on regret matching
            total_positive_regret = sum(max(0, r) for r in self.regret_sum.values())
            for move in range(9):
                if total_positive_regret > 0:
                    self.strategy[move] = max(0, self.regret_sum[move]) / total_positive_regret
                else:
                    self.strategy[move] = 1 / 9  # Uniform strategy if no positive regrets
        return self.strategy


class Game:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 0
        self.win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 0

    def apply_move(self, move):
        if self.board[move] == ' ':
            self.board[move] = 'X' if self.current_player == 0 else 'O'
            #print(self.board)
            return True
        return False

    def check_winner(self):
        # Check for a winner
        for pos in self.win_positions:
            if self.board[pos[0]] == self.board[pos[1]] == self.board[pos[2]] != ' ':
                if self.board[pos[0]] == "X":
                    #print("Player 0 Wins ----------------")
                    return True, (1, -1)  # Player 0 wins
                else:
                    #print("Player 1 Win ----------------")
                    return True, (-1, 1)  # Player 1 wins

        # Check for a tie
        counter = 0
        for pos in self.win_positions:
            line = {self.board[p] for p in pos}
            if 'X' in line and 'O' in line:
                counter += 1
        if counter == 8:
            #print("Tie")
            return True, (0,0)
        # Game is not over
        return False, (0, 0)

class Player:
    def __init__(self, algo, gamma=None, player_num = 0):
        self.algo = algo  # "mmcts" or "mccfr"
        self.gamma = gamma  # Only for MMCTS 
        self.player_num = player_num
        self.infoset = "|"
        self.root = Node(self.player_num, "|", in_tree=True)
        self.current_node = self.root
        self.tree_mode = True
        self.last_node_before_tree_mode = self.root

    def reset(self):
        self.current_node = self.root
        self.infoset = "|"
        self.tree_mode = True
        self.last_node_before_tree_mode = self.root

    def update_tree(self, move, prob):
        # Update the tree based on the selected algorithm
        if self.algo == "mmcts":
            self.mmcts_tree_update(move, prob)
        elif self.algo == "mccfr":
            self.mccfr_tree_update(move)

    def mmcts_tree_update(self, move, prob = 0):
        if move not in self.current_node.children:
            new_infoset= self.current_node.infoset+str(move)
            new_node = Node(infoset = new_infoset,player = self.player_num,parent=self.current_node, depth=self.current_node.depth + 1, in_tree = False)
            self.current_node.children[move] = new_node

            if self.tree_mode == True:
                self.last_node_before_tree_mode = new_node
                self.tree_mode = False
        
        else:
            if self.tree_mode == True and self.current_node.children[move].in_tree == False:
                self.tree_mode = False
                self.last_node_before_tree_mode = self.current_node.children[move]

        self.current_node = self.current_node.children[move]
        self.current_node.probability = prob
        self.infoset = self.current_node.infoset
        #print("node ", self.infoset, " updated with prob", prob)

    def mmcts_update(self, reward, player):
        #print("now updating -------------------------------")
        if self.tree_mode:
            self.last_node_before_tree_mode = self.current_node
        self.last_node_before_tree_mode.in_tree = True
        while self.last_node_before_tree_mode != None:

            depth_temp = self.last_node_before_tree_mode.depth
            move_temp = self.last_node_before_tree_mode.move
            prob_temp = self.last_node_before_tree_mode.probability
            scaling_factor = np.exp(max(min(5,1.7**(depth_temp - 8)* reward / prob_temp),-5))
            #print("infoset ", self.last_node_before_tree_mode.infoset, " depth ",depth_temp," move ", move_temp, " prob from parent ",prob_temp)

            if player == 0:
                if self.last_node_before_tree_mode.infoset not in infoset_index_zero:
                    #print("terminal node encountered !!!!!!!!!!!!")
                    break
                infoset_idx = infoset_index_zero[self.last_node_before_tree_mode.infoset]
                #print("rewards:", reward_zero[infoset_idx], "before: ", reward_zero[infoset_idx][move_temp])
                reward_zero[infoset_idx][move_temp] = reward_zero[infoset_idx][move_temp]*scaling_factor
                reward_zero[infoset_idx] = reward_zero[infoset_idx]/np.sum(reward_zero[infoset_idx])
                #print("scaling exp(1.7**depth-7)*reward/prob ", scaling_factor, "after: ",reward_zero[infoset_idx][move_temp])
            else:
                if self.last_node_before_tree_mode.infoset not in infoset_index_one:
                    #print("terminal node encountered !!!!!!!!!!!!")
                    break
                infoset_idx = infoset_index_one[self.last_node_before_tree_mode.infoset]
                #print("rewards:", reward_one[infoset_idx], "before: ", reward_one[infoset_idx][move_temp])
                reward_one[infoset_idx][move_temp] = reward_one[infoset_idx][move_temp]*scaling_factor
                reward_one[infoset_idx] = reward_one[infoset_idx]/np.sum(reward_one[infoset_idx])
                #print("scaling exp(1.7**depth-7)*reward/prob ", scaling_factor, "after: ",reward_one[infoset_idx][move_temp])
           
            self.last_node_before_tree_mode = self.last_node_before_tree_mode.parent
        
    def mccfr_update(self, move, reward):
        if move not in self.current_node.children:
            new_node = Node(parent=self.current_node, move=move, depth=self.current_node.depth + 1)
            self.current_node.children[move] = new_node
        # Backpropagate regrets
        for action, strat in self.current_node.strategy.items():
            counterfactual_value = reward if action == move else 0
            regret = counterfactual_value - strat * reward
            self.current_node.regret_sum[action] += regret

        for action in range(9):
            self.current_node.strategy_sum[action] += self.current_node.strategy[action]

game = Game()
player0 = Player(algo="mmcts", player_num = 0)
player1 = Player(algo="mmcts", player_num = 1)

iterations = 120000000  # Time in seconds

for idx in tqdm(range(iterations)):
    game.reset()
    player0.reset()
    player1.reset()

    while True:
        active_player = player0 if game.current_player == 0 else player1
        move, prob = active_player.current_node.sample_action(algo ="mmcts", gamma = (idx+1)**(-0.3))
        outcome = game.apply_move(move)
        move = str(move)+"*" if outcome else str(move)+"." 
        active_player.update_tree(move, prob)
        finished, reward = game.check_winner()
        if finished:
            #print("player 0 ",player0.infoset, player0.tree_mode, player0.last_node_before_tree_mode.infoset)
            #print("player 1 ",player1.infoset, player1.tree_mode, player1.last_node_before_tree_mode.infoset)
            break
        else:
            game.current_player = 1 - game.current_player  # Switch player


    if reward[0] != 0:
        player0.mmcts_update(reward[0], player = 0)
        player1.mmcts_update(reward[1], player = 1)
    
    if idx % 1200000 == 0:
        name0 = 'mcts_pttt_pl0' + '_' + str(idx) + '.npy'
        name1 = 'mcts_pttt_pl1' + '_' + str(idx) + '.npy'
        np.save(name0, reward_zero)
        np.save(name1, reward_one)  

