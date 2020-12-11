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

# height, eroded, r trans, c trans, holes, wells, hole depth, row hole
h = -27
e = 27
r_t = -25
c_t = 14
h_t = -23
w = 0
h_d = -16
r_h = -16

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
    
MAX_WEIGHT = 30
MIN_WEIGHT = -30
FEATURES = 8
class GeneticAgent:
    
    def __init__(self, pop_count=20, cutoff=1000):
        self.pop = self.population(pop_count)
        self.cutoff = cutoff
        self.best_fs = np.zeros(FEATURES)
        self.evolutions = 1
    
    #parse a previously generated feature set
    #to avoid rerunning the evolution
    def load(self, data):
        for i in range(data.len):
            self.best_fs[i] = data[i]
       
    #format the feature weights to dump into txt file
    def save(self):
        features = open("features.txt", "w")
        for feature in self.best_fs:
            features.write(str(feature) + '\n')
    
    #create a feature set with random weights
    def individual(self):
        fs = []
        for i in range(FEATURES):
            fs.append(random.randint(MIN_WEIGHT, MAX_WEIGHT))
        return fs
    
    #create a population of feature sets with pop_count members
    def population(self, pop_count):
        pop = []
        for i in range(pop_count):
            pop.append(self.individual())
        return pop
    
    #determine the fitness of a feature set (score obtained using this fs)
    #have the agent simulate play using this feature set and retain the score
    def fitness(self, fs, start_state):
        drops = 0
        state = start_state
        self.actions = Queue()
        while not state.lost and drops <= self.cutoff:
            state = state.do_action(self.move(state, fs))
            drops += 1
        print(fs)
        if state.score > 0:
            print(state.score + drops)
            return state.score + drops
        else:
            print(drops)
            return drops
    
    def train(self, state):
        self.evolve(state)
    
    #run one iteration of evolution for this agent's feature set
    #save the best one
    def evolve(self, start_state, retain = 0.2, random_select = 0.05, mutate = 0.1):
        #determine fitness of each individual, sort them by fitness, then
        #get the individuals we will use to reproduce
        graded = [(self.fitness(ind, start_state), ind) for ind in self.pop]
        graded = [ind[1] for ind in sorted(graded)]
        retain_length = int(len(graded) * retain)
        parents = graded[(len(graded) - retain_length):]
        self.best_fs = parents[len(parents) - 1]
        print(self.best_fs)
            
        #randomly add some worse-performing individuals
        for ind in graded[:(len(graded) - retain_length)]:
            if random_select > random.random():
                parents.append(ind)
                    
        #mutate some individuals, picking a random new weight for one feature
        for ind in parents:
            if mutate > random.random():
                mutate_feature = random.randint(0, FEATURES - 1)
                ind[mutate_feature] = random.randint(MIN_WEIGHT, MAX_WEIGHT)
                    
                    
        #crossover parents to generate children
        parents_length = len(parents)
        desired_length = len(self.pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male) // 2
                child = male[:half] + female[half:]
                children.append(child)

        parents.extend(children)
        self.pop = parents
        print('evolution ' + str(self.evolutions) + ' finished!')
        self.evolutions += 1

    #hard drop the piece once it is above the best state
    def move(self, state, weights=[]):
        if self.actions.empty() and len(weights) == 0:
            self.search(state, self.best_fs)
        elif self.actions.empty():
            self.search(state, weights)
        else:
            action = self.actions.get()
            return action

    
    def search(self, state, weights):
        rotations = 4
        #square does not rotate
        if state.current.shape == SHAPES[3]:
            rotations = 1
        #s, z, and l pieces rotate once
        elif (state.current.shape == SHAPES[0]
            or state.current.shape == SHAPES[1]
            or state.current.shape == SHAPES[2]):
            rotations = 2
        #save each rotation of the current piece
        valid_rotations = []
        rotate_state = copy.deepcopy(state)
        for rot in range(rotations):
            rotate_state = rotate_state.do_action(state.ROTATE)
            valid_rotations.append(rotate_state)
        #iterate through the shape rotations and check each location
        max_score = float('-inf')
        best_state = state
        current_rotation = 0
        best_rotation = 0
        for state in valid_rotations:
            state_right = copy.deepcopy(state)
            state_left = copy.deepcopy(state)
            while state_right.current.x < COLS:
                prev_x = state_right.current.x
                state_right = state_right.do_action(state.RIGHT)
                #if we've moved as far right as possible, break
                if prev_x == state_right.current.x:
                    break
                #otherwise check if the state resulting from a hard drop is better
                #than our best state seen so far
                score = self.calculate_score(state_right.result().get_eval_score(), weights)
                if score > max_score:
                    max_score = score
                    best_state = state_right
                    best_rotation = current_rotation
            while state_left.current.x > 0:
                prev_x = state_left.current.x
                state_left = state_left.do_action(state.LEFT)
                #if we've moved as far left as possible, break
                if prev_x == state_left.current.x:
                    break
                #otherwise check if the state resulting from a hard drop is better
                #than our best state seen so far
                score = self.calculate_score(state_left.result().get_eval_score(), weights)
                if score > max_score:
                    max_score = score
                    best_state = state_left
                    best_rotation = current_rotation
            current_rotation += 1
        
        #save the sequence of actions for this piece
        current_rotation = 0
        while current_rotation <= best_rotation:
            state = state.do_action(state.ROTATE)
            self.actions.put(state.ROTATE)
            current_rotation += 1
        while state.current.x != best_state.current.x:
            if state.current.x < best_state.current.x:
                state = state.do_action(state.RIGHT)
                self.actions.put(state.RIGHT)
            elif state.current.x > best_state.current.x:
                state = state.do_action(state.LEFT)
                self.actions.put(state.LEFT)
            else:
                break
        #end with a hard drop
        self.actions.put(state.HARD_DROP)


    def get_score_after_moves(self, state, actions, weights):
        for elem in list(actions.queue):
            state = state.do_action(elem)
        result = state.result()
        return self.calculate_score(result.get_eval_score(), weights)


    def calculate_score(self, score, weights):
        weighted_score = []

        for f in range(len(score)):
            weighted_score.append(score[f] * weights[f])

        return sum(weighted_score)
        