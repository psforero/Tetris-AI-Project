import pygame # only for the human agent
import torch
from torch import nn
from torch import optim
import numpy as np
from queue import Queue
import copy

COLS = 10
ROWS = 20

LEFT = 1
RIGHT = 2
DOWN = 3
ROTATE = 4
HARD_DROP = 5

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


class NaiveAgent:

    def __init__(self):
        self.weights = []

    def train(self, state):
        actions = 0
        while not state.lost:
            actions += 1
            state = state.do_action(state.DOWN)
        
        self.weights.append(actions)
        
    def move(self, state):
        return state.DOWN
    
    def save(self):
        return self.weights
    
    def load(self, data):
        print(data)
        self.weights = data





# height, eroded, r trans, c trans, holes, wells, hole depth, row hole
h = -1.0
e = 2.0
r_t = -1.0
c_t = -1.0
h_t = -4.0
w = -1.4
h_d = -1.0
r_h = -1.5


class HandTunedAgent:
    def __init__(self):
        self.rotations = 0
        self.actions = Queue()

    def search(self, state, n):
        score = self.calculate_score(state.get_eval_score())
        max_action = HARD_DROP

        for action in range(1, 4):
            new_state = state.do_action(action)
            if new_state.piece_num > state.piece_num or n == 0:
                return (score, max_action)
            (best_score, best_action) = self.search(new_state, n-1)
            if best_score > score:
                score = best_score
                max_action = action
        
        return (score, max_action)


    def move(self, state):
        _, action = self.search(state, 3)
        return action

    # def move(self, state):

    #     if self.actions.empty():
    #         self.search(state)
    #         if self.actions.empty():
    #             return DOWN
    #         return self.actions.get()

    # def search(self, state):

    #     max_score = float('-inf')
    #     rotations = 0
    #     side_moves = 0

    #     # for side in range(6):
    #     #     result = state.result()
    #     #     score = self.calculate_score(result.get_eval_score())

    #     #     if score > max_score:
    #     #         max_score = score
    #     #         side_moves = side
    #     #         print(score, max_score, side_moves)
    #     #     state = state.do_action(ROTATE)

    #     # for i in range(side_moves):
    #     #     self.actions.put(LEFT)

    #     next_state = copy.deepcopy(state)
    #     while next_state.valid_space(next_state.current, next_state.grid):
    #         next_state = next_state.do_action(LEFT)


    #     for rotation in range(len(state.current.rotation)):
    #         result = state.result()
    #         score = self.calculate_score(result.get_eval_score())
        
    #         if score > max_score:
    #             max_score = score
    #             rotations = rotation
    #             print(score, max_score, rots)
    #         state = state.do_action(ROTATE)
        
    #     for i in range(rotations):
    #         self.actions.put(ROTATE)


    def calculate_score(self, score):
        weights = [h, e, r_t, c_t, h_t, w, h_d, r_h]
        weighted_score = []

        for f in range(len(score)):
            weighted_score.append(score[f] * weights[f])

        return sum(weighted_score)

def density(state):
    total = 0
    filled = 0
    for i in range(len(state.grid) - 1, -1, -1):
        row = state.grid[i]
        has_pieces = False
        for j in range(len(row)):
            total += 1
            if (j, i) in state.locked:
                has_pieces = True
                filled += 1
        if not has_pieces:
            break
    return total - filled

class NNAgent:
    def __init__(self):
        self.n_inputs = 12
        self.n_outputs = 4
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 8), 
            nn.ReLU(),  
            nn.Linear(8, self.n_outputs),
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
        # cost function results + current piece
        nn_input = []
        
        for k in state.get_eval_score():
            nn_input.append(k)

        # for y, x in state.locked.keys():
        #     nn_input[10*y + x] = 1
        
        nn_input.append(state.current.x)
        nn_input.append(state.current.y)
        nn_input.append(SHAPES.index(state.current.shape))
        nn_input.append(state.current.rotation)
        return np.array(nn_input)

    def train(self, state):
        gamma = 0.99
        batch_size = 20
        epsilon = 1.0

        init_state = state
        states = []
        rewards = []
        actions = []

        while not state.lost:
            last_state = state
            state = state.do_action(state.DOWN)

            # Get actions and convert to numpy array
            action_probs = self.predict(self.vectorize(state)).detach().numpy()

            # prioritize exploration early in each batch, decreasing later
            if (np.random < 1 - self.batch_counter * .05):
                action = np.random.choice(self.action_space)
            else:
                action = np.random.choice(self.action_space, p=action_probs)

            state = state.do_action(action + 1)
            done = state.lost

            costs = state.get_eval_score()
            if density(state) < density(last_state):
                density_r = 10
            else:
                density_r = -10

            r = last_state.score - state.score + density_r
            
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
                
    def train_old(self, state):
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
            if density(state) < density(last_state):
                density_r = 10
            else:
                density_r = -10

            r = last_state.score - state.score # + density_r
            
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
        action_probs = self.predict(self.vectorize(state)).detach().numpy()
        print(action_probs)
        action = np.random.choice(self.action_space, p=action_probs)
        return action

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