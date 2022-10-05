import random
import math


BOT_NAME = "INSERT NAME FOR YOUR BOT HERE OR IT WILL THROW AN EXCEPTION" #+ 19 


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        if state.is_full():
            finals = state.scores()
            return finals[0] - finals[1]
        if state.next_player() == 1:
            best_val = -math.inf
            for i in state.successors():
                val = self.minimax(i[1])
                best_val = max(val, best_val)
            return best_val
        else:
            best_val = math.inf
            for i in state.successors():
                val = self.minimax(i[1])
                best_val = min(val, best_val)
            return best_val



class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        count = -1
        for row in state.get_rows():
            for elem in row:
                if elem == 1 or elem == -1:
                    count += 1
        if self.depth_limit == count:
            return self.evaluation(state)
        if state.is_full():
            finals = state.scores()
            return finals[0] - finals[1]
        if state.next_player() == 1:
            best_val = -math.inf
            for i in state.successors():
                val = self.minimax(i[1])
                best_val = max(val, best_val)
                return best_val
        else:
            best_val = math.inf
            for i in state.successors():
                val = self.minimax(i[1])
                best_val = min(val, best_val)
                return best_val

    def get_streaks(self, line):
        xtot = 0
        otot = 0
        xcount = 0
        ocount = 0
        for row in line:
            for elem in row:
                if elem == 1 and xcount >= 1:
                    xcount += 1
                elif elem == 0 and xcount >= 2:
                    xtot += 1
                else:
                    xcount = 0
                if elem == -1 and ocount >= 1:
                    ocount += 1
                elif elem == 0 and ocount >= 2:
                    xtot += 1
                else:
                    ocount = 0
        return xtot, otot


    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        xtot = 0
        otot = 0
        rows = self.get_streaks(state.get_rows())
        cols = self.get_streaks(state.get_cols())
        diags = self.get_streaks(state.get_diags())
        xtot += rows[0] + cols[0] + diags[0]
        otot += rows[1] + cols[1] + diags[1]
        return otot - xtot


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""


    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """

        def alpha_beta(st, alpha, beta):
            if st.is_full():
                finals = st.scores()
                return finals[0] - finals[1]
            if st.next_player() == 1:
                best_val = -math.inf
                for i in st.successors():
                    val = alpha_beta(i[1], alpha, beta)
                    best_val = max(val, best_val)
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break
                return best_val
            else:
                best_val = math.inf
                for i in st.successors():
                    val = alpha_beta(i[1], alpha, beta)
                    best_val = min(val, best_val)
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break
                return best_val

        return alpha_beta(state, -math.inf, math.inf)  # Change this line!


# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.

