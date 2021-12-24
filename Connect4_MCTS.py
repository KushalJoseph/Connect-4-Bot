# IMPORTS AND GLOBAL VARIABLES:
import numpy as np
import random

height, width = 6, 7
c = 1.414
num_trials = 1
win_reward, loss_reward, draw_reward = 1, -1, 0.5
num_games = 30

###############################################################################################

# GAME/BOARD FUNCTIONS:
def next_player(cur_player):
    if(cur_player == 1):
        return 2
    elif(cur_player == 2):
        return 1

def make_move(state, move):
    '''
        Given a state and a number 0-4 indicating which column a move was played and player, 
        the player's number, this function returns the New State after the move is played.
    '''
    cur_player = check_next_player(state)

    new_state = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(state[i][j])
        new_state.append(row)
        

    # If this row is already full, return the same state back
    if(new_state[0][move] != 0):
        return state

    for i in range(0, height):
        if(new_state[height-i-1][move] == 0):
            new_state[height-i-1][move] = cur_player
            break

    return new_state

def check_won(state, player):
    '''
        Given a state and a player (1 or 2), check whether that player has won the game
    '''
    for i in range(0, height):
        for j in range(0, width):
            if(state[i][j] == player):
                if(i >= 3):
                    if(state[i-1][j] == player and state[i-2][j] == player and state[i-3][j] == player):
                        return True

                if(i <= height - 4):
                    if(state[i+1][j] == player and state[i+2][j] == player and state[i+3][j] == player):
                        return True

                if(j >= 3):
                    if(state[i][j-1] == player and state[i][j-2] == player and state[i][j-3] == player):
                        return True

                if(j <= width - 4):
                    if(state[i][j+1] == player and state[i][j+2] == player and state[i][j+3] == player):
                        return True

                if(i <= (height - 4) and j <= (width - 4)):
                    if(state[i+1][j+1] == player and state[i+2][j+2] == player and state[i+3][j+3] == player):
                        return True

                if(i >= 3 and j >= 3):
                    if(state[i-1][j-1] == player and state[i-2][j-2] == player and state[i-3][j-3] == player):
                        return True

                if(i >= 3 and j <= (width - 4)):
                    if(state[i-1][j+1] == player and state[i-2][j+2] == player and state[i-3][j+3] == player):
                        return True

                if(i <= (height - 4) and j >= 3):
                    if(state[i+1][j-1] == player and state[i+2][j-2] == player and state[i+3][j-3] == player):
                        return True

    return False

def check_draw(state):
    '''
        Given a state, check if no player has won and all the holes have been filled
    '''
    if(check_won(state, 1) or check_won(state, 2)):
        return False

    for i in range(0, height):
        for j in range(0, width):
            if(state[i][j] == 0):
                return False

    return True

def check_game_over(state):
    '''
        Given a state, determine whether a player has won the game or the game is tied, i.e, 
        no player has won but all the holes in the gameboard have been filled
    '''
    return (check_won(state, 1) or check_won(state, 2) or check_draw(state))

def legalMoves(state):
    '''
        Given a state, this function returns a List of legal Moves (list of integers)
        Any integer 0-(width - 1) not on this list is illegal because that column is already complete
    '''
    legalmoves = []
    for i in range(width):
        if(state[0][i] == 0):
            legalmoves.append(i)
    return legalmoves

def check_next_player(state):
    '''
        To find out the next player, all we need is the state. Because we can count the number
        of 1 and 2 coins already on the board
    '''
    one_count = two_count = 0
    for i in range(height):
        for j in range(width):
            if(state[i][j] == 1):
                one_count += 1
            elif(state[i][j] == 2):
                two_count += 1

    if(one_count == two_count):
        return 1
    else:
        return 2


################################################################################################
# MCTS NODE CLASS AND FUNCTIONS:

player = None

class MCTSNode:
    def __init__(self, state, parent):
        self.state = state
        self.visit_count = 0
        self.reward = 0
        self.children = {} # Dictionary mapping move (0-4) to the state resulting from that move
        self.parent = parent 
        
    # Returns true if a given Node is not fully expanded along all "width" moves
    def isNotFullyExpanded(self):
        if(len(self.children.keys()) < width):
            return True
        else:
            return False
    
    # Adds a child to this node at 'index'
    def addChild(self, child_node, index):
        self.children[index] = child_node


def MC(state, num_playouts):

    '''
        Returns an integer 0-(width - 1), the OPTIMAL MC move given a state and the Number of Playouts
    '''

    global player

    options = []
    for f in range(0, num_trials):
        root = MCTSNode(state, parent = None)
        for i in range(0, num_playouts):    
            player = check_next_player(state)

            selected_state = selection(root)
                
            reward = simulate(selected_state.state)

            backprop(selected_state, reward)
            

        # print("\nFINISHED:")
        # print("Printing all children of root with their associated rewards:")
        # for i in range(width):
        #     if(root.children.get(i, -1) != -1):
        #         print(f"Child at {i} has reward = {root.children[i].reward} and numVisits = {root.children[i].visit_count}")
        #         PrintGrid(root.children[i].state)
        #     else:
        #         print(f'The root doesnt have children with move {i}')
               
        best_child_index = bestChild(root, c = 0) # The action leading to the best state (bestChild() returns this)
        options.append(best_child_index)
        # print(f"The best child is move: {best_child_index} because its reward is {root.children[best_child_index].reward} and numVisits = {root.children[best_child_index].visit_count}")
        # print("And the state resulting from that is:")
        # PrintGrid(root.children[best_child_index].state)
        # # print("OVER")

    ret = max(options, key = options.count)
    return ret


def selection(node):
    '''
        MCTS Selection Function
    '''
    
    while(check_game_over(node.state) == False):
        if(node.isNotFullyExpanded()):
            return expand(node)
        
        else: 
            node = node.children[bestChild(node)]
            
    return node
    

def expand(node):
    '''
        MCTS Expand Function
    '''
    legalmoves = legalMoves(node.state)
    for move in legalmoves:
        if(node.children.get(move, -1) == -1):    # Node with move "i" has not been created yet
            # This is an untried action
            new_state = make_move(node.state, move)
            
            child_node = MCTSNode(new_state, parent = node)
            node.children[move] = child_node
            return node.children[move]

    return (node.children[bestChild(node)])

        
def bestChild(node, c = c):
    '''
        Returns the best child of a given node for MCTS
        If c is given as 0 (we are finally selecting the best child of root), we only look at the 
        visit_count of each child
    '''
    best_val = -100000.0
    
    values = []
    for i in range(width):
        cur_child = node.children.get(i, -1)
        if(cur_child == -1):
            values.append(None)
            continue
          
        cur_val = 0
        if(c == 0):
            cur_val = cur_child.visit_count
        else:
            cur_val = (cur_child.reward/cur_child.visit_count)
            cur_val += c * np.sqrt(np.log(float(node.visit_count))/float(cur_child.visit_count))

        values.append(cur_val)

        if(cur_val > best_val):
            best_val = cur_val


    best_children = []
    for i in range(width):
        if(values[i] == best_val):
            best_children.append(i)
    
    return random.choice(best_children)


def simulate(state):
    '''
        Simulates (Randomly) the outcome from a particular state, and returns the reward
    '''
    while(check_game_over(state) == False):
        action = np.random.randint(0, width)
        k = make_move(state, action)
        
        if(k == state):
            continue
        
        state = k
    
    reward = 0
    if(check_won(state, player) == True):
        reward = win_reward
    elif(check_won(state, next_player(player)) == True):
        reward = loss_reward
    elif(check_draw(state) == True):
        reward = draw_reward
    
    return reward


def backprop(node, reward):
    '''
        On the basis of reward received, this function updates the visit_count and rewards of all
        nodes visited in this playout
    '''
    global player
    while(node is not None):
        cur_player = check_next_player(node.state)
        
        node.visit_count += 1
        node.reward += -reward if (cur_player == player) else (reward)
        
        # print(f"The current node is: and its reward is incremented by {reward} so it is now = {node.reward}")
        # PrintGrid(node.state)
    
        node = node.parent

  
# MCTS Game playing functions:
def play_one_mc40_vs_mc200(firstplayer):
    init_board = [
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0],
                 ]
    
    player = 1
    board = init_board
    while(check_game_over(board) == False):
        PrintGrid(board)
        if(player == 1): 
            if(firstplayer == "40"):
                board = make_move(board, MC(init_board, 10))
            elif(firstplayer == "200"):
                board = make_move(board, MC(init_board, 400))
                
            # board = k
            player = 2
            
        elif(player == 2):
            if(firstplayer == "40"):
                board = make_move(board, MC(init_board, 400))
            elif(firstplayer == "200"):
                board = make_move(board, MC(init_board, 10))

            # board = k
            player = 1
           
    winner = 0
    PrintGrid(board)
    if(check_won(board, 1) == True):
        winner = 1
    elif(check_won(board, 2) == True):
        winner = 2
        
    if(winner == 0):
        print("Game resulted in a draw")
        return "draw"
    else:
        if(firstplayer == "40"):
            if(winner == 1):
                print("MC40 won this game")
                return "40"
            else:
                print("MC200 won this game")
                return "200"
        elif(firstplayer == "200"):
            if(winner == 2):
                print("MC40 won this game")
                return "40"
            else:
                print("MC200 won this game")
                return "200"
    

def play_multiple_mc40_vs_mc200(num_games):
    mc40_wins, mc200_wins, ties = 0, 0, 0
    
    for i in range(num_games // 2):
        print(f'Playing game {i * 2}')
        result = play_one_mc40_vs_mc200(firstplayer = "40")
        if(result == "40"):
            mc40_wins += 1
        elif(result == "200"):
            mc200_wins += 1
        else:
            ties += 1

        print(f'Playing game {i * 2 + 1 }')
        result = play_one_mc40_vs_mc200(firstplayer = "200")
        if(result == "40"):
            mc40_wins += 1
        elif(result == "200"):
            mc200_wins += 1
        else:
            ties += 1
    
    if(i % 2 != 0):
        result = play_one_mc40_vs_mc200(firstplayer = "200")
        if(result == "40"):
            mc40_wins += 1
        elif(result == "200"):
            mc200_wins += 1
        else:
            ties += 1   
            
    
    print(f"{num_games} were played")
    print(f"MC40 won {mc40_wins} times while MC200 won {mc200_wins} times and they tied {ties} times.")

   






#########################################################################################

def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    
    for i in range(width):
        print('_', end = " ")
    print()
    for i in range(width):
        print(f'{i}', end = " ")
    print()
    print()
    
def main(): 
    
    while(True):  
        print("Welcome to Connect-4. I am a bot that uses the Monte-Carlo Tree Search Algorithm to play against you!")

        print("Press 1 to play first or 2 if you want me to play first")

        k = int(input())

        while(k != 1 and k != 2):
            print("Please Press either 1 or 2. 1 if you want to play first, else 2")
            k = int(input())

        board = [
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],
                ]

        if(k == 1):
            turn = "player"
        elif(k == 2):
            turn = "mc"

        while(check_game_over(board) == False):
            if(turn == "player"):
                print("Press a number between 0 and 6 (both included), indicating which column you want to place your piece")
                print("Notice the column numbers written below to assist you")
                player_move = int(input())

                while(player_move >= width):
                    print(f"Please enter a Column number to drop your piece, between 0 and {width - 1}")
                    player_move = int(input())

                legalmoves = legalMoves(board)
                while(player_move not in legalmoves):
                    print("Your move was not allowed since that column is already full. Please try another move")
                    player_move = int(input())
                    
                board = make_move(board, player_move)

                turn = "mc"

            elif(turn == "mc"):
                board = make_move(board, MC(board, 250))

                turn = "player"

            PrintGrid(board)

        if((k == 1 and check_won(board, 1)) or (k == 2 and check_won(board, 2))):
            print("Congrats, you Won!")
        elif(k == 1 and check_won(board, 2) or (k == 2 and check_won(board, 1))):
            print("Sorry. You got defeated by the MCTS Algorithm!")
        else:
            print("Match resulted in a draw")

        print("Press 1 to play again. Press any other key to Quit")

        play_again = input()
        if(int(play_again) != 1):
            print("Thanks for playing!")
            break
    
    


    
if __name__ == '__main__':
    main()