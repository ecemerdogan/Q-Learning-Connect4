import openpyxl
from env import State
from agents import ComputerPlayer, HumanPlayer
import time
import os

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
sheet['D1'] = "Fail/Success/Draw"
sheet['E1'] = "Number of moves made by Computer"

# training
roundNum =80000
p1 = ComputerPlayer("p1")
p2 = ComputerPlayer("p2")
st = State(p1, p2)
print("training...")
st.train(roundNum)
p1.savePolicy()

# play with human
p1 = ComputerPlayer("computer", exp_rate=0)
p1.loadPolicy("policy_p1")
p2 = HumanPlayer("human")
st = State(p1, p2)

# Record the current time before executing the code
start_time = time.time()
st.play_against_human()

# Record the current time after the game is over.
end_time = time.time()
elapsed_time = end_time - start_time

# Load the rows count from a file if it exists
rows_file = "rows_count.txt"
if os.path.exists(rows_file):
    with open(rows_file, "r") as f:
        rows = int(f.read().strip()) +1
else:
    rows = 1
    
winnerOfTheGame = st.win
if winnerOfTheGame == 1:
    result = "S"
elif winnerOfTheGame == -1:
    result = "F"
elif winnerOfTheGame == 0:
    result = "D"

#Excel Table to Test
nextRow = rows + 1
print(nextRow)
sheet['A' + str(nextRow)] = p1.lr
sheet['B' + str(nextRow)] = p1.exp_rate 
sheet['C' + str(nextRow)] = roundNum
sheet['D' + str(nextRow)] = result
sheet['E' + str(nextRow)] = st.number_of_move
workbook.save("data.xlsx")
# Save the updated rows count to the file
with open(rows_file, "w") as f:
    f.write(str(rows))
    