from template import Agent
import random
import time
import math
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque

# class myAgent(Agent):
# def __init__(self,_id):
# super().__init__(_id)

# def SelectAction(self,actions,game_state):
# return random.choice(actions)


PLAYER_COUNT = 2
INF = 10000
THINKTIME = 0.9


class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.other_id = (self.id + 1) % PLAYER_COUNT
        self.mcts = MCTS(self.id, 12)

    def SelectAction(self, actions, game_state):
        return self.mcts.choose_move(game_state)


class Node():
    def __init__(self, state, id, parent=None, move=None, branch_limit=3):
        self.state = state
        self.parent = parent
        self.move = move
        self.id = id
        self.win_counts = {None: 0, 0: 0, 1: 0}
        self.num_rollouts = 0
        self.children = []

        simulator = GameRule(PLAYER_COUNT)
        actions = simulator.getLegalActions(state, id)

        if len(actions) <= branch_limit:
            self.unvisited_moves = actions
        else:
            list = []
            for action in actions:
                next_state = simulator.generateSuccessor(deepcopy(state), action, id)
                list.append((next_state, action))

            reward_list = [(reward(next_state, id), action) for next_state, action in list]
            sorted_reward_list = sorted(reward_list, key=lambda x: x[0], reverse=True)[:branch_limit]
            self.unvisited_moves = [action for _, action in sorted_reward_list]


class MCTS:
    def __init__(self, agent_id, num_rollouts, exploration_weight=1):
        self.num_rollouts = num_rollouts
        self.exploration_weight = exploration_weight
        self.agent_id = agent_id

    def choose_move(self, game_state):
        start_time = time.time()

        for i in range(self.num_rollouts):
            if self.time_exceeded(start_time):
                break

            node = Node(game_state, self.agent_id)
            node = self.select_node(node)

            if len(node.unvisited_moves) != 0:
                new_node = self.expand_node(node)
                winner = self.simulate_random_game(new_node)
                self.backpropagate(node, winner)

        return self.select_best_move(game_state)

    def time_exceeded(self, start_time):
        return THINKTIME <= time.time() - start_time

    def select_node(self, node):
        # Selection phase: Traverse down the tree until an unvisited move is found or no more children exist
        while len(node.unvisited_moves) == 0 and node.children:
            node = self.select_child(node)
        return node

    def expand_node(self, node):
        # Expansion phase: Expand by selecting an unvisited move
        simulator = GameRule(PLAYER_COUNT)
        move = node.unvisited_moves[0]
        node.unvisited_moves.remove(move)

        next_state = deepcopy(node.state)
        simulator.current_agent_index = node.id
        simulator.current_game_state = simulator.generateSuccessor(next_state, move, node.id)
        simulator.current_agent_index = simulator.getNextAgentIndex()

        new_node = Node(simulator.current_game_state, simulator.current_agent_index, node, move)
        node.children.append(new_node)

        return new_node

    def backpropagate(self, node, winner):
        # Backpropagation phase: Update the stats for all nodes on the path back to the root
        while node is not None:
            node.win_counts[winner] += 1
            node.num_rollouts += 1
            node = node.parent

    def select_best_move(self, game_state):
        # Final phase: Select the best move based on the win rate
        best_reward = -1
        best_move = None

        for child in Node(game_state, self.agent_id).children:
            if child.num_rollouts != 0:
                win_rate = child.win_counts[self.agent_id] / child.num_rollouts
            else:
                win_rate = 0

            if win_rate > best_reward:
                best_reward = win_rate
                best_move = child.move

        return best_move





    def select_child(self, node: Node):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None

        if len(node.children) == 1:
            return node.children[0]

        for child in node.children:
            score = self.ucb1(total_rollouts, child.num_rollouts,
                              child.win_counts[node.id], child.win_counts[None])
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def choose_move(self, game_state):
        start_time = time.time()

        for i in range(self.num_rollouts):
            if self.time_exceeded(start_time):
                break

            node = Node(game_state, self.agent_id)
            node = self.select_node(node)

            if len(node.unvisited_moves) != 0:
                new_node = self.expand_node(node)
                winner = self.simulate_random_game(new_node)
                self.backpropagate(node, winner)

        return self.select_best_move(game_state)

    def time_exceeded(self, start_time):
        return THINKTIME <= time.time() - start_time

    def select_node(self, node):
        # Selection phase: Traverse down the tree until an unvisited move is found or no more children exist
        while len(node.unvisited_moves) == 0 and node.children:
            node = self.select_child(node)
        return node

    def expand_node(self, node):
        # Expansion phase: Expand by selecting an unvisited move
        simulator = GameRule(PLAYER_COUNT)
        move = node.unvisited_moves[0]
        node.unvisited_moves.remove(move)

        next_state = deepcopy(node.state)
        simulator.current_agent_index = node.id
        simulator.current_game_state = simulator.generateSuccessor(next_state, move, node.id)
        simulator.current_agent_index = simulator.getNextAgentIndex()

        new_node = Node(simulator.current_game_state, simulator.current_agent_index, node, move)
        node.children.append(new_node)

        return new_node

    def backpropagate(self, node, winner):
        # Backpropagation phase: Update the stats for all nodes on the path back to the root
        while node is not None:
            node.win_counts[winner] += 1
            node.num_rollouts += 1
            node = node.parent

    def select_best_move(self, game_state):
        # Final phase: Select the best move based on the win rate
        best_reward = -1
        best_move = None

        for child in Node(game_state, self.agent_id).children:
            if child.num_rollouts != 0:
                win_rate = child.win_counts[self.agent_id] / child.num_rollouts
            else:
                win_rate = 0

            if win_rate > best_reward:
                best_reward = win_rate
                best_move = child.move

        return best_move

    def simulate_random_game(self, node: Node, depth=4):
        simulator = GameRule(PLAYER_COUNT)
        state = node.state

        next_state = deepcopy(state)
        simulator.current_game_state = next_state
        simulator.current_agent_index = node.id

        for i in range(depth):
            if simulator.current_agent_index == PLAYER_COUNT:
                break

            actions = simulator.getLegalActions(simulator.current_game_state, simulator.current_agent_index)
            if len(actions) == 1:
                action = actions[0]
            else:
                action = get_best_action(simulator.current_game_state,
                                         actions,
                                         simulator.current_agent_index)

            simulator.current_game_state = simulator.generateSuccessor(next_state,
                                                                       action,
                                                                       simulator.current_agent_index)
            simulator.current_agent_index = simulator.getNextAgentIndex()

        other_id = (self.agent_id + 1) % PLAYER_COUNT
        choice1 = reward(simulator.current_game_state, self.agent_id)
        choice2 = reward(simulator.current_game_state, other_id)

        if choice1 > choice2:
            return self.agent_id
        elif choice1 < choice2:
            return other_id
        else:
            return None

    def ucb1(self, total_rollouts, node_rollouts, win_count, draw_count):
        if node_rollouts == 0:
            return float('inf')

        win_rate = win_count / node_rollouts
        exploration_factor = math.sqrt(math.log(total_rollouts) / node_rollouts)

        ucb_result = win_rate + self.exploration_weight * exploration_factor
        return ucb_result


def reward(state, id):
    agent_state = state.agents[id]
    reward_value = 0

    reward_value += grid_reward(agent_state.grid_state)
    reward_value += first_agent_bonus(state, id)
    reward_value += score_round_bonus(agent_state)
    reward_value += end_of_game_bonus(agent_state)

    return reward_value


def grid_reward(grid_state):
    reward_value = 0
    for x in range(5):
        for y in range(5):
            reward_value += evaluate_grid_position(x, y, grid_state[x][y])
    return reward_value


def evaluate_grid_position(x, y, grid_value):
    if x == 2 and y == 2 and grid_value == 1:
        return 2
    elif 0 < x < 4 and 0 < y < 4 and grid_value == 1:
        return 1
    return 0


def first_agent_bonus(state, id):
    if state.next_first_agent == id:
        return 1
    return 0


def score_round_bonus(agent_state):
    return agent_state.ScoreRound()[0]


def end_of_game_bonus(agent_state):
    return agent_state.EndOfGameScore()


def get_best_action(state, actions, id):
    start_time = time.time()
    simulator = GameRule(PLAYER_COUNT)

    best_reward = -float('inf')
    best_action = random.choice(actions)

    for action in actions:
        if time_exceeded(start_time):
            break

        next_state = simulate_next_state(state, simulator, action, id)
        reward_value = compute_reward(next_state, id)

        best_reward, best_action = update_best_action(best_reward, best_action, reward_value, action)

    return best_action


#
def time_exceeded(start_time):
    return time.time() - start_time >= THINKTIME


def simulate_next_state(state, simulator, action, id):
    next_state = deepcopy(state)
    return simulator.generateSuccessor(next_state, action, id)


def compute_reward(state, id):
    return reward(state, id)


def update_best_action(best_reward, best_action, reward_value, action):
    if reward_value > best_reward:
        return reward_value, action
    return best_reward, best_action
   
