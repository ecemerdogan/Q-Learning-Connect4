# Reinforcement Learning Connect 4 Game with Epsilon Greedy Policy.
# Each player has twenty-one discs.
import numpy as np
import os


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
        self.win = None
        self.number_of_move=0
        
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
    def train(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                #If you do not need to train the agent, remove the if block from the comment, 
                #to play the game with the existing policy.
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
    def play_against_human(self):
        # Let's satrt with p1.
        self.current_player = self.p1
        while not self.isEnd:
            positions = self.availablePositions()
            if self.current_player== self.p1:
                self.number_of_move+=1
                action = self.current_player.chooseAction(positions, self.board, self.playerSymbol)
            else:
                action = self.current_player.chooseAction(positions)
            
            self.updateState(action)
            self.showBoard()
            self.win = self.winner()
            if self.win is not None:
                if self.win == 1 or self.win==-1:
                    print(self.current_player.name, "wins!")
                
                if self.win == 0:
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
