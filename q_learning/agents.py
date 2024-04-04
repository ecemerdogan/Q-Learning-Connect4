from env import BOARD_COLS,BOARD_ROWS
import math
import numpy as np
import pickle
import os

#ComputerPlayer class represents our agent.
#Recording and updating states-values after each game.
class ComputerPlayer:
    #ϵ-greedy method is used to balance exploration and exploitation.
    def __init__(self, name, exp_rate=1.0, min_exp_rate=0.01):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.3 # learning rate.
        self.exp_rate = exp_rate #ϵ: exp_rate.
        self.min_exp_rate = min_exp_rate #Minimum exploration rate
        self.decay_gamma = 0.9 # discount factor.
        self.states_value = {}  # state -> value

    # With hash structure, we will store the actions in the 1D vector format which is converted to the string, then.
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= math.exp(-self.exp_rate):
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -math.inf
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p

        # print("{} takes action {}".format(self.name, action))
        # Decay exploration rate
        self.exp_rate = max(self.min_exp_rate, self.exp_rate * self.decay_gamma)
        return action

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        file_path = 'policy_' + str(self.name)
        print("Saving policy to:", file_path)
        try:
            with open(file_path, 'wb') as fw:
                pickle.dump(self.states_value, fw)
            print("Policy saved successfully.")
        except Exception as e:
            print(f"Error saving policy: {e}")

    def loadPolicy(self, file):
        file_path = os.path.join(os.getcwd(), file)
        try:
            with open(file_path, 'rb') as fr:
                self.states_value = pickle.load(fr)
        except FileNotFoundError:
            print(f"Error: File {file} not found.")

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))

            action = (row, col)
            if action in positions:
                return action
            else:
                print("Invalid action. Please choose a valid position.")
    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass               
