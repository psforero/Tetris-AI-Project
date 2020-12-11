import pygame # only for the human agent
import torch
from torch import nn
from torch import optim
import numpy as np
from queue import Queue
import copy
import random

COLS = 10
ROWS = 20

LEFT = 1
RIGHT = 2
DOWN = 3
ROTATE = 4
HARD_DROP = 5

MOVES = [LEFT, RIGHT, ROTATE, HARD_DROP]

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

SHAPES = [S, Z, I, O, J, L, T]


class RandomAgent:

    def move(self, state):
        return random.choice(MOVES)



# height, eroded, r trans, c trans, holes, wells, hole depth, row hole
height = -1.8
eroded = 8.0
r_trans = -0.6
c_trans = -0.6
holes = -8.0
wells = -1.5
h_depth = -2.2
r_hole = -0.8


class HandTunedAgent:
    def __init__(self):
        self.actions = Queue()

    def move(self, state):

        if not self.actions.empty():
            return self.actions.get()

        self.search(state)

        #return HARD_DROP

    def search(self, state):

        # Check left
        state_left = copy.deepcopy(state)
        max_score_left = self.get_score_after_moves(state_left, self.actions)
        rotations_left = 0
        moves_left = 0
        for side in range(6):
            # Check rotation
            for rotation in range(len(state_left.current.shape)):
                result = state_left.result()
                score = self.calculate_score(result.get_eval_score())

                if score > max_score_left:
                    max_score_left = score
                    rotations_left = rotation
                    moves_left = side

                state_left = state_left.do_action(ROTATE)
            state_left = state_left.do_action(LEFT)

        # Check right
        state_right = copy.deepcopy(state)
        max_score_right = self.get_score_after_moves(state_right, self.actions)
        rotations_right = 0
        moves_right = 0

        for side in range(5):
            # Check rotation
            for rotation in range(len(state_right.current.shape)):
                result = state_right.result()
                score = self.calculate_score(result.get_eval_score())

                if score > max_score_right:
                    max_score_right = score
                    rotations_right = rotation
                    moves_right = side

                state_right = state_right.do_action(ROTATE)
            state_right = state_right.do_action(RIGHT)

        if max_score_left > max_score_right:
            for i in range(rotations_left):
                self.actions.put(ROTATE)

            for i in range(moves_left):
                self.actions.put(LEFT)
        else:
            for i in range(rotations_right):
                self.actions.put(ROTATE)

            for i in range(moves_right):
                self.actions.put(RIGHT)

        self.actions.put(HARD_DROP)

    def get_score_after_moves(self, state, actions):
        for elem in list(actions.queue):
            state = state.do_action(elem)
        result = state.result()
        return self.calculate_score(result.get_eval_score())

    def calculate_score(self, score):
        weights = [height, eroded, r_trans, c_trans, holes, wells, h_depth, r_hole]
        weighted_score = []

        for f in range(len(score)):
            weighted_score.append(score[f] * weights[f])

        return sum(weighted_score)


class NNAgent:
    def __init__(self):
        self.n_inputs = 204
        self.n_outputs = 4

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))

        self.total_rewards = []
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1

        # Define optimizer
        self.optimizer = optim.Adam(self.network.parameters(),
                           lr=0.01)

        self.action_space = np.arange(4)

    def predict(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def vectorize(self, state):
        # 10x20 grid + current piece
        nn_input = np.zeros(200 + 4)

        for y, x in state.locked.keys():
            nn_input[10*y + x] = 1

        nn_input[200] = state.current.x
        nn_input[201] = state.current.y
        nn_input[202] = SHAPES.index(state.current.shape)
        nn_input[203] = state.current.rotation
        return np.array(nn_input)

    def train(self, state):
        gamma = 0.99
        batch_size = 10

        init_state = state
        states = []
        rewards = []
        actions = []

        while not state.lost:
            last_state = state
            state = state.do_action(state.DOWN)

            # Get actions and convert to numpy array
            action_probs = self.predict(self.vectorize(state)).detach().numpy()
            action = np.random.choice(self.action_space, p=action_probs)

            state = state.do_action(action + 1)
            done = state.lost

            costs = state.get_eval_score()

            r = last_state.score - state.score - sum(costs)

            states.append(self.vectorize(last_state))
            rewards.append(r)
            actions.append(action)

            # If done, batch data
            if done:
                self.batch_rewards.extend(self.discount_rewards(rewards, gamma))
                self.batch_states.extend(states)
                self.batch_actions.extend(actions)
                self.batch_counter += 1
                self.total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if self.batch_counter == batch_size:
                    self.optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(self.batch_states)
                    reward_tensor = torch.FloatTensor(self.batch_rewards)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor([self.batch_actions])
                    # Calculate loss
                    logprob = torch.log(self.predict(state_tensor))

                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()

                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    self.optimizer.step()

                    self.batch_rewards = []
                    self.batch_actions = []
                    self.batch_states = []
                    self.batch_counter = 1

                avg_rewards = np.mean(self.total_rewards[-100:])
                # Print running average
                print("\rAverage of last 100: {:.2f}".format(avg_rewards), end="")

    def move(self, state):
        outputs = self.predict(self.vectorize(state))
        # print(outputs)
        return torch.argmax(self.predict(self.vectorize(state))).detach() + 1

    def save(self):
        return self.network.state_dict()

    def load(self, data):
        self.network.load_state_dict(data)
        self.network.eval()


class HumanAgent:

    def move(self, state):
        action = None

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = state.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = state.RIGHT
                elif event.key == pygame.K_DOWN:
                    action = state.DOWN
                elif event.key == pygame.K_UP:
                    action = state.ROTATE
                elif event.key == pygame.K_SPACE:
                    action = state.HARD_DROP

        return action