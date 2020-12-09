import pygame # only for the human agent
import random
from GUI import GameState

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
    
MAX_WEIGHT = 50
MIN_WEIGHT = 0
FEATURES = 8
class GeneticAgent:
    
    #parse a previously generated feature set
    #to avoid rerunning the evolution
    def load_featureset(self):
        features = open("features.txt", "r")
        lines = features.readlines()
        fs = []
        for line in lines:
            fs.append(int(line))
        return fs
        
    #create a feature set with random weights
    def individual(self):
        fs = []
        for i in range(FEATURES):
            fs.append(random.randint(MIN_WEIGHT, MAX_WEIGHT))
        return fs
    
    #create a population of feature sets with pop_count members
    def population(self, pop_count):
        self.pop = []
        for i in range(pop_count):
            self.pop.append(self.individual())
    
    #determine the fitness of a feature set (score obtained using this fs)
    #have the agent play using this feature set and retain the score
    def fitness(self, fs, cutoff = 100):
        return sum(fs)
    
    #run one iteration of evolution for this agent's feature set
    def evolve(self, retain = 0.2, random_select = 0.05, mutate = 0.01):
            #determine fitness of each individual, sort them by fitness, then
            #get the individuals we will use to reproduce
            graded = [(self.fitness(ind), ind) for ind in self.pop]
            graded = [ind[1] for ind in sorted(graded)]
            retain_length = int(len(graded) * retain)
            parents = graded[retain_length:]
            
            #randomly add some worse-performing individuals
            for ind in graded[retain_length:]:
                if random_select > random.random():
                    parents.append(ind)
                    
            #mutate some individuals, replacing a feature weight with a random new one
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
    
    #run the genetic algorithm to obtain an optimal featureset
    def optimize_featureset(self, iterations = 100, pop_count = 10):
        self.population(pop_count)
        print(sum(self.pop[0]))
        for i in range(iterations):
            self.evolve()
        print(sum(self.pop[0]))
        
    #simulate playing the game using a given featureset
    def play(self, featureset):     
        return
        
agent = GeneticAgent()
agent.optimize_featureset()