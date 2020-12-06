import pygame
import random
import copy

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per blo ck
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height


# SHAPE FORMATS

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

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape


class Piece(object):
    rows = 20  # y
    columns = 10  # x

    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


class GameState:
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    ROTATE = 4

    def __init__(self, locked_positions, current_piece, next_piece):
        self.locked = locked_positions
        self.current = current_piece
        self.next = next_piece
        self.grid = create_grid(locked_positions)
        self.lost = False

    def do_action(self, action):
        new_state = copy.deepcopy(self)
        new_state.grid = create_grid(new_state.locked)

        if action == self.LEFT:
            new_state.current.x -= 1
            if not valid_space(new_state.current, new_state.grid):
                new_state.current.x += 1
        elif action == self.RIGHT:
            new_state.current.x += 1
            if not valid_space(new_state.current, new_state.grid):
                new_state.current.x -= 1
        elif action == self.ROTATE:
            new_state.current.rotation = new_state.current.rotation + 1 % len(new_state.current.shape)
            if not valid_space(new_state.current, new_state.grid):
                new_state.current.rotation = new_state.current.rotation - 1 % len(new_state.current.shape)
        elif action == self.DOWN:
            new_state.current.y += 1
                
            # if hit bottom, change piece
            if not valid_space(new_state.current, new_state.grid):
                new_state.current.y -= 1
                new_state.locked = lock_shape(new_state.locked, new_state.current)
                new_state.locked = clear_rows(new_state.grid, new_state.locked)
                new_state.grid = create_grid(new_state.locked)
                
                new_state.current = new_state.next
                new_state.next = get_random_shape()

        if check_lost(new_state.locked):
            new_state.lost = True

        return new_state


def create_grid(locked_positions={}):
    grid = [[(0,0,0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j,i) in locked_positions:
                c = locked_positions[(j,i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_random_shape():
    global shapes, shape_colors

    return Piece(5, 0, random.choice(shapes))


def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width() / 2), top_left_y + play_height/2 - label.get_height()/2))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128,128,128), (sx, sy+ i*30), (sx + play_width, sy + i * 30))  # horizontal lines
        for j in range(col):
            pygame.draw.line(surface, (128,128,128), (sx + j * 30, sy), (sx + j * 30, sy + play_height))  # vertical lines


def clear_rows(grid, locked):
    new_locked = {}
    locked_i = len(grid)-1

    for i in range(len(grid)-1,-1,-1):
        row = grid[i]
        all_locked = True
        for j in range(len(row)):
            if (j, i) not in locked:
                all_locked = False

        if not all_locked:
            for j in range(len(row)):
                if (j, i) in locked:
                    new_locked[(j, locked_i)] = locked[(j, i)]
            locked_i -= 1

    return new_locked

def lock_shape(locked, piece):
    shape_pos = convert_shape_format(piece)
    for pos in shape_pos:
        p = (pos[0], pos[1])
        locked[p] = piece.color
    return locked


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*30, sy + i*30, 30, 30), 0)

    surface.blit(label, (sx + 10, sy- 30))


def draw_window(surface, grid):
    surface.fill((0,0,0))
    # Tetris Title
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('TETRIS', 1, (255,255,255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 30))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j* 30, top_left_y + i * 30, 30, 30), 0)

    # draw grid and border
    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 5)

def main():
    state = GameState(
        locked_positions = {}, # (x,y):(255,0,0)
        current_piece = get_random_shape(),
        next_piece = get_random_shape()
    )

    run = True

    clock = pygame.time.Clock()
    fall_time = 0

    while run:
        fall_speed = 0.27
        fall_time += clock.get_rawtime()
        clock.tick()

        # PIECE FALLING CODE
        if fall_time/1000 >= fall_speed:
            fall_time = 0
            state = state.do_action(GameState.DOWN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    state = state.do_action(GameState.LEFT)
                elif event.key == pygame.K_RIGHT:
                    state = state.do_action(GameState.RIGHT)
                elif event.key == pygame.K_UP:
                    state = state.do_action(GameState.ROTATE)
                elif event.key == pygame.K_DOWN:
                    state = state.do_action(GameState.DOWN)

        grid = state.grid

        shape_pos = convert_shape_format(state.current)

        # add current piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = state.current.color

        draw_window(win, grid)
        draw_next_shape(state.next, win)
        pygame.display.update()

        # Check if user lost
        if state.lost:
            run = False

    draw_text_middle("You Lost", 40, (255,255,255), win)
    pygame.display.update()
    pygame.time.delay(2000)


def main_menu():
    run = True
    while run:
        win.fill((0,0,0))
        draw_text_middle('Press any key to begin.', 60, (255, 255, 255), win)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                main()
    pygame.quit()


win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')

main_menu()  # start game