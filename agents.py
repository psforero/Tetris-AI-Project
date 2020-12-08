import pygame # only for the human agent

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