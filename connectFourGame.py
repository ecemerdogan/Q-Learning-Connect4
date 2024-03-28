# Reinforcement Learning Connect 4 Game with Epsilon Greedy Policy.
# Each player has twenty-one discs.
import numpy as np
import time
import pickle
import os
import math 
import openpyxl

print("Current Working Directory:", os.getcwd)

BOARD_ROWS = 6
BOARD_COLS = 7

# This class will be acting as board and judger in the game.
class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        #Initialization of players.
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        
        # When a player takes an action playerSymbol will be filled in the board and the board state will be updated.
        # init p1 plays first
        self.playerSymbol = 1

    # get unique hash of current board state
    # The getHash function hashes the current board state so that it can be stored in the state-value dictionary.
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash
    
    #After each action taken by players, check the winner if the game is ended.
    #Return 1 --> Computer wins
    #Return -1 --> Human wins.
    def winner(self):
    # Check rows for a winner
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS - 3):
                if (self.board[i, j] == 1 and self.board[i, j + 1] == 1 and self.board[i, j + 2] == 1 and self.board[i, j + 3] == 1):
                    self.isEnd = True
                    return 1
                elif (self.board[i, j] == -1 and self.board[i, j + 1] == -1 and self.board[i, j + 2] == -1 and self.board[i, j + 3] == -1):
                    self.isEnd = True
                    return -1
        
        # Check columns for a winner
        for i in range(BOARD_ROWS - 3):
            for j in range(BOARD_COLS):
                if (self.board[i, j] == 1 and self.board[i + 1, j] == 1 and self.board[i + 2, j] == 1 and self.board[i + 3, j] == 1):
                    self.isEnd = True
                    return 1
                elif (self.board[i, j] == -1 and self.board[i + 1, j] == -1 and self.board[i + 2, j] == -1 and self.board[i + 3, j] == -1):
                    self.isEnd = True
                    return -1
        
        # Check diagonals for a winner (positive slope)
        for i in range(BOARD_ROWS - 3):
            for j in range(BOARD_COLS - 3):
                if (self.board[i, j] == 1 and self.board[i + 1, j + 1] == 1 and self.board[i + 2, j + 2] == 1 and self.board[i + 3, j + 3] == 1):
                    self.isEnd = True
                    return 1
                elif (self.board[i, j] == -1 and self.board[i + 1, j + 1] == -1 and self.board[i + 2, j + 2] == -1 and self.board[i + 3, j + 3] == -1):
                    self.isEnd = True
                    return -1
        
        # Check diagonals for a winner (negative slope)
        for i in range(3, BOARD_ROWS):
            for j in range(BOARD_COLS - 3):
                if (self.board[i, j] == 1 and self.board[i - 1, j + 1] == 1 and self.board[i - 2, j + 2] == 1 and self.board[i - 3, j + 3] == 1):
                    self.isEnd = True
                    return 1
                elif (self.board[i, j] == -1 and self.board[i - 1, j + 1] == -1 and self.board[i - 2, j + 2] == -1 and self.board[i - 3, j + 3] == -1):
                    self.isEnd = True
                    return -1

        # Check for a tie
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        
        self.isEnd = False
        return None
       
    def availablePositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    if (i == BOARD_ROWS - 1 or (i!= BOARD_ROWS-1 and self.board[i+1,j]!=0)):
                        positions.append((i, j))  # need to be tuple
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))     
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1

    # play for train the AI
    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                if os.path.exists("policy_p1"):
                    print("Policy has been already saved!")
                    break
                print("Rounds {}".format(i))
            self.current_player = self.p1
            while not self.isEnd:
                positions = self.availablePositions()
                action_player = self.current_player.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(action_player)
                board_hash = self.getHash()
                self.p1.addState(board_hash)
                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

    # play with human
    def play2(self):
        # Başlangıçta p1 ile başlayalım
        self.current_player = self.p1
        while not self.isEnd:
            positions = self.availablePositions()
            if self.current_player== self.p1:
                action = self.current_player.chooseAction(positions, self.board, self.playerSymbol)
            else:
                action = self.current_player.chooseAction(positions)
            
            self.updateState(action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.current_player.name, "wins!")
                elif win == -1:
                    print(self.current_player.name, "wins!")
                else:
                    print("tie!")
                self.current_player.reset()
                self.reset()
                return
            # Bir sonraki oyuncuya geçiş yapalım
            self.current_player = self.p2 if self.current_player == self.p1 else self.p1


    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'B'
                if self.board[i, j] == -1:
                    token = 'R'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('------------------------------')

#Player class represents our agent.
#Recording ands updating states-values after each game.
class Player:
    #ϵ-greedy method is used to balance exploration and exploitation.
    def __init__(self, name, exp_rate=1.0, min_exp_rate=0.01):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.1 # learning rate.
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
                # print("value", value)
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

if __name__ == "__main__":

    # Record the current time before executing the code
    start_time = time.time()

    # Check if the file exists
    file_path = "data.xlsx"
    if os.path.exists(file_path):
        # If the file exists, open it
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
    else:
        # If the file doesn't exist, create a new workbook and sheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        # Add headers to the sheet
        sheet['A1'] = "Learning Rate"
        sheet['B1'] = "Exploration Rate"
        sheet['C1'] = "Round Num"
        sheet['D1'] = "Fail/Success"
        sheet['E1'] = "Training Process in Sec."

    # training
    roundNum =60000
    p1 = Player("p1")
    p2 = Player("p2")
    st = State(p1, p2)
    print("training...")
    st.play(roundNum)
    p1.savePolicy()

    # Record the current time after training process is over.
    end_time = time.time()
    elapsed_time = end_time - start_time

    # play with human
    p1 = Player("computer", exp_rate=0)
    p1.loadPolicy("policy_p1")
    p2 = HumanPlayer("human")
    st = State(p1, p2)
    st.play2()

    # Load the rows count from a file if it exists
    rows_file = "rows_count.txt"
    if os.path.exists(rows_file):
        with open(rows_file, "r") as f:
            rows = int(f.read().strip())
    else:
        rows = 0

    #Excel Table to Test
    nextRow = rows + 1
    print(nextRow)
    sheet['A' + str(nextRow)] = p1.lr
    sheet['B' + str(nextRow)] = p1.exp_rate 
    sheet['C' + str(nextRow)] = roundNum
 
    sheet['D' + str(nextRow)] = elapsed_time
    workbook.save("data.xlsx")

    # Save the updated rows count to the file
    with open(rows_file, "w") as f:
        f.write(str(rows))
    
