import sys
import puzz
import pdqpq
from random import choice
from functools import reduce


MAX_SEARCH_ITERS = 100000
GOAL_STATE = puzz.EightPuzzleBoard("012345678")

def get_cost(state, direction):
    empty_spot = state.find("0")
    if direction == 'up':
        return int(state._get_tile(empty_spot[0], empty_spot[1] + 1)) ** 2
    if direction == 'down':
        return int(state._get_tile(empty_spot[0], empty_spot[1] - 1)) ** 2
    if direction == 'left':
        return int(state._get_tile(empty_spot[0] - 1, empty_spot[1])) ** 2
    if direction == 'right':
        return int(state._get_tile(empty_spot[0] + 1, empty_spot[1])) ** 2

def get_total_cost(state, direction, parent_cost):
    empty_spot = state.find("0")
    if direction == 'up':
        return parent_cost + int(state._get_tile(empty_spot[0], empty_spot[1] + 1)) ** 2
    if direction == 'down':
        return parent_cost + int(state._get_tile(empty_spot[0], empty_spot[1] - 1)) ** 2
    if direction == 'left':
        return parent_cost + int(state._get_tile(empty_spot[0] - 1, empty_spot[1])) ** 2
    if direction == 'right':
        return parent_cost + int(state._get_tile(empty_spot[0] + 1, empty_spot[1])) ** 2

def get_misplaced(state):
    count = 0
    for i in range(len(state._board)):
        if state._board[i] != i:
            count += 1
    return count

def get_manhattan(state):
    total = 0
    for i in range(len(state._board)):
        if state._board[i] != i:
            length = abs(int(state._board[i]) - i)
            if length == 3 or 1:
                total += 1
            elif length == 4 or 2 or 6:
                total += 2
            elif length == 5 or 7 or 8:
                total += 3
    return total

def get_special_manhattan(state):
    total = 0
    for i in range(len(state._board)):
        if state._board[i] != i:
            length = abs(int(state._board[i]) - i)
            if length == 3 or 1:
                total += 1 * (int(state._board[i]) ** 2)
            elif length == 4 or 2 or 6:
                total += 2 * (int(state._board[i]) ** 2)
            elif length == 5 or 7 or 8:
                total += 3 * (int(state._board[i]) ** 2)
    return total


def bfs(start_state, results):
    if(start_state == GOAL_STATE):
        return
    goal = 0
    frontier = []
    explored = []
    frontier.append(start_state)
    tree = {start_state: "none"}
    results['frontier_count'] += 1
    while len(frontier) != 0:
        if not isinstance(goal, int):
            break
        node = frontier.pop(0)
        explored.append(node)
        results['expanded_count'] += 1
        succs = node.successors()
        for n in succs:
            successor = succs[n]
            if (successor not in frontier) and (successor not in explored):
                if successor == GOAL_STATE:
                    goal = successor
                    tree[goal] = node
                    break
                else:
                    frontier.append(successor)
                    tree[successor] = node
                    results['frontier_count'] += 1
    curr_state = goal
    while curr_state != start_state:
        succs = tree[curr_state].successors()
        for n in succs:
            if succs[n] == curr_state:
                results['path'].insert(0, (n, curr_state))
                results['path_cost'] += get_cost(curr_state, n)
        curr_state = tree[curr_state]
    return results

def search(start_state, results, heuristic, type):
    if (start_state == GOAL_STATE):
        return
    goal = 0
    frontier = pdqpq.PriorityQueue()
    explored = []
    tree = {start_state: "none"}
    frontier.add(start_state, 0)
    results['frontier_count'] += 1
    while len(frontier) != 0:
        node_cost = frontier.peek()
        while node_cost == None:
            node_cost = frontier.peek()
        node_cost = node_cost[1]
        node = frontier.pop()
        if (node == GOAL_STATE):
            goal = node
            break
        explored.append(node)
        results['expanded_count'] += 1
        succs = node.successors()
        for n in succs:
            successor = succs[n]
            in_frontier = frontier.__contains__(successor)
            if type == "ucost":
                cost = get_total_cost(successor, n, node_cost)
            elif type == "greedy":
                cost = heuristic(successor)
            elif type == "astar":
                cost = heuristic(successor) + get_total_cost(successor, n, node_cost)
            if (not in_frontier) and (successor not in explored):
                frontier.add(successor, cost)
                tree[successor] = node
                results['frontier_count'] += 1
            elif in_frontier and (frontier.get(successor) > cost):
                frontier.add(successor, cost)
                tree[successor] = node
    curr_state = goal
    while curr_state != start_state:
        succs = tree[curr_state].successors()
        for n in succs:
            if succs[n] == curr_state:
                results['path'].insert(0, (n, curr_state))
                results['path_cost'] += get_cost(curr_state, n)
        curr_state = tree[curr_state]
    return results


def solve_puzzle(start_state, strategy):
    results = {
        'path': [],
        'path_cost': 0,
        'frontier_count': 0,
        'expanded_count': 0,
    }
    if strategy == "bfs":
        print("")
        #bfs(start_state, results)
    elif strategy == "ucost":
        print("")
        #search(start_state, results, get_misplaced, "ucost")
    elif strategy == "greedy-h1":
        search(start_state, results, get_misplaced, "greedy")
    elif strategy == "greedy-h2":
        search(start_state, results, get_manhattan, "greedy")
    elif strategy == "greedy-h3":
        search(start_state, results, get_special_manhattan, "greedy")
    elif strategy == "astar-h1":
        print("")
        ##search(start_state, results, get_manhattan, "astar")
    elif strategy == "astar-h3":
        print("")
        #search(start_state, results, get_special_manhattan, "astar")
    return results

def print_summary(results):
    if 'path' in results:
        print("found solution of length {}, cost {}".format(len(results['path']),
                                                            results['path_cost']))
        for move, state in results['path']:
            print("  {:5} {}".format(move, state))
    else:
        print("no solution found")
    print("{} states placed on frontier, {} states expanded".format(results['frontier_count'],
                                                                    results['expanded_count']))

if __name__ == '__main__':
    board = [[i * 3 + j for j in range(3)] for i in range(3)]
    x, y = 0, 0
    for _ in range(int(sys.argv[3])):
        possibilities = []
        if x in [1, 2]:
            possibilities.append([-1, 0])
        if x in [0, 1]:
            possibilities.append([1, 0])
        if y in [1, 2]:
            possibilities.append([0, -1])
        if y in [0, 1]:
            possibilities.append([0, 1])

        move = choice(possibilities)
        newX, newY = x + move[0], y + move[1]
        board[x][y] = board[newX][newY]
        board[newX][newY] = 0
        x, y = newX, newY
    print(''.join([''.join([str(c) for c in row]) for row in board]))

    start = puzz.EightPuzzleBoard(''.join([''.join([str(c) for c in row]) for row in board]))
    method = sys.argv[2]

    print("solving puzzle {} -> {}".format(start, GOAL_STATE))
    results = solve_puzzle(start, method)
    print_summary(results)
