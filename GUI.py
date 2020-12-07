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
S_WIDTH = 800
S_HEIGHT = 700
PLAY_WIDTH = 300  # meaning 300 // 10 = 30 width per block
PLAY_HEIGHT = 600  # meaning 600 // 20 = 20 height per blo ck
BLOCK_SZ = 30
COLS = 10
ROWS = 20

TOP_LEFT_X = (S_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = S_HEIGHT - PLAY_HEIGHT

# COLORS
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

# TETRAMINOS

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
SHAPE_COLORS = [GREEN, RED, CYAN, YELLOW, ORANGE, BLUE, PURPLE]

# index 0 - 6 represent shape


"""
///////////// GAME LOGIC ////////////////
"""


class Piece(object):
    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = SHAPE_COLORS[SHAPES.index(shape)]
        self.rotation = 0  # number from 0-3


class GameState:
    LEFT = 1
    RIGHT = 2
    DOWN = 3
    ROTATE = 4
    HARD_DROP = 5

    def __init__(self):
        self.locked = {}
        self.current = get_shape()
        self.next = get_shape()
        self.grid = create_grid(self.locked)
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
            if not valid_space(new_state.current, new_state.grid):
                self.hit_bottom(new_state)

        elif action == self.HARD_DROP:
            while valid_space(new_state.current, new_state.grid):
                new_state.current.y += 1
            self.hit_bottom(new_state)

        if check_lost(new_state.locked):
            new_state.lost = True

        return new_state

    def result(self):
        result_state = copy.deepcopy(self)
        result_piece = self.get_result_piece(result_state)

        result_state.locked = lock_shape(result_state.locked, result_piece)
        shape_pos = convert_shape_format(result_piece)

        # add piece's resulting position to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                result_state.grid[y][x] = result_piece.color

        return result_state

    def hit_bottom(self, new_state):
        new_state.current.y -= 1
        new_state.locked = lock_shape(new_state.locked, new_state.current)
        new_state.locked = clear_rows(new_state.grid, new_state.locked)
        new_state.grid = create_grid(new_state.locked)
        new_state.current = new_state.next
        new_state.next = get_shape()

    def get_result_piece(self, result_state):
        result_piece = copy.deepcopy(self.current)
        result_piece.color = GRAY

        while valid_space(result_piece, result_state.grid):
            result_piece.y += 1
        result_piece.y -= 1

        return result_piece


def create_grid(locked_positions={}):
    grid = [[BLACK for col in range(COLS)] for row in range(ROWS)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid


def get_shape():
    return Piece(COLS // 2, 0, random.choice(SHAPES))


def lock_shape(locked, piece):
    shape_pos = convert_shape_format(piece)
    for pos in shape_pos:
        p = (pos[0], pos[1])
        locked[p] = piece.color
    return locked


def convert_shape_format(piece):
    positions = []
    format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(COLS) if grid[i][j] == BLACK] for i in range(ROWS)]
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


def clear_rows(grid, locked):
    new_locked = {}
    locked_i = len(grid) - 1

    for i in range(len(grid) - 1, -1, -1):
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


"""
///////////// COST FUNCTION ////////////////
"""


def get_eval_score(state):
    score = [landing_height(state),
             eroded(state),
             row_transitions(state),
             col_transitions(state),
             holes(state),
             well_depth(state),
             hole_depth(state),
             row_holes(state)]

    return score


def landing_height(state):  # changed to resulting height
    j = state.get_result_piece(state).x
    height = 0
    for i in range(len(state.grid), -1, -1):
        if (j, i) in state.locked:
            height = i

    return ROWS - height


def eroded(state):  # missing * "contribution per piece"
    lines_cleared = 0

    for i in range(len(state.grid) - 1, -1, -1):
        row = state.grid[i]
        all_locked = True
        for j in range(len(row)):
            if (j, i) not in state.locked:
                all_locked = False

        if all_locked:
            lines_cleared += 1

    return lines_cleared


def row_transitions(state): # first col not counted?
    transitions = 0

    for i in range(len(state.grid) - 1, -1, -1):
        row = state.grid[i]

        for j in range(len(row) - 1):
            if (((j, i) in state.locked and (j + 1, i) not in state.locked) or
                    ((j, i) not in state.locked and (j + 1, i) in state.locked)):
                transitions += 1

    return transitions


def col_transitions(state):
    transitions = 0

    for j in range(len(state.grid[0])):
        for i in range(len(state.grid) - 1, -1, -1):
            if (((j, i) in state.locked and (j, i - 1) not in state.locked) or
                    ((j, i) not in state.locked and (j, i - 1) in state.locked)):
                transitions += 1

    return transitions


def holes(state):
    holes = 0

    for i in range(len(state.grid) - 1, -1, -1):
        row = state.grid[i]
        for j in range(len(row)):
            if (j, i) not in state.locked:
                for c in range(i - 1, -1, -1):
                    if (j, c) in state.locked:
                        holes += 1
                        break
    return holes


def well_depth(state):
    heights = [0 for i in range(COLS)]

    for j in range(len(state.grid[0])):
        for i in range(len(state.grid) - 1, 0, -1):
            if (j, i) in state.locked or i == 0:
                heights[j] = ROWS-i

    wells = [0 for i in range(COLS)]
    for i in range(len(heights)):
        difference = 0
        if i == 0:
            if heights[i+1] > heights[i]:
                difference = heights[i+1] - heights[i]
        elif i == COLS-1:
            if heights[i-1] > heights[i]:
                difference = heights[i-1] - heights[i]
        else:
            if heights[i-1] > heights[i] and heights[i+1] > heights[i]:
                difference = min(heights[i+1] - heights[i], heights[i-1] - heights[i])
        wells[i] = difference

    for w in range(len(wells)):
        total = 0
        for val in range(wells[w] + 1):
            total += val
        wells[w] = total

    return sum(wells)


def hole_depth(state):
    cumulative_holes = 0

    for j in range(len(state.grid[0])):
        for i in range(len(state.grid) - 1, -1, -1):
            if (j, i) not in state.locked:
                depth = 0
                for c in range(i - 1, -1, -1):
                    if (j, c) in state.locked:
                        depth += 1
                    else:
                        cumulative_holes += depth
                        break

    return cumulative_holes


def row_holes(state):
    row_holes = 0

    for i in range(len(state.grid) - 1, -1, -1):
        row = state.grid[i]
        row_found = False
        for j in range(len(row)):
            if (j, i) not in state.locked:
                for c in range(i - 1, -1, -1):
                    if (j, c) in state.locked and not row_found:
                        row_holes += 1
                        row_found = True
                        break

    return row_holes


"""
///////////// DRAWING FUNCTIONS ////////////////
"""


def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, True, color)

    surface.blit(label, (
        TOP_LEFT_X + PLAY_WIDTH / 2 - (label.get_width() / 2), TOP_LEFT_Y + PLAY_HEIGHT / 2 - label.get_height() / 2))


def draw_grid_lines(surface):
    sx = TOP_LEFT_X
    sy = TOP_LEFT_Y
    for i in range(ROWS):
        pygame.draw.line(surface, GRAY, (sx, sy + i * BLOCK_SZ),
                         (sx + PLAY_WIDTH, sy + i * BLOCK_SZ))  # horizontal lines
        for j in range(COLS):
            pygame.draw.line(surface, GRAY, (sx + j * BLOCK_SZ, sy),
                             (sx + j * BLOCK_SZ, sy + PLAY_HEIGHT))  # vertical lines


def draw_next_shape(piece, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', True, WHITE)

    sx = TOP_LEFT_X + PLAY_WIDTH + 50
    sy = TOP_LEFT_Y + PLAY_HEIGHT // 2 - 100
    format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (sx + j * BLOCK_SZ, sy + i * BLOCK_SZ, BLOCK_SZ, BLOCK_SZ))

    surface.blit(label, (sx + 10, sy - 30))


def draw_eval_score(score, surface):
    score_labels = ["landing h (l): ",
                    "erosion (e): ",
                    "row trans (delta r): ",
                    "col trans (delta c): ",
                    "holes (L): ",
                    "wells (W)",
                    "hole depth (D)",
                    "row holes (R)"]

    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('COST FUNCTION', True, WHITE)

    sx = 10
    sy = TOP_LEFT_Y
    surface.blit(label, (sx + 10, sy - 30))

    for i in range(len(score_labels)):
        sy += 25
        label = font.render(score_labels[i] + str(score[i]), True, WHITE)
        surface.blit(label, (sx + 10, sy - 30))


def draw_window(surface, grid):
    surface.fill(BLACK)

    # Tetris Title
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('TETRIS', True, WHITE)

    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH / 2 - (label.get_width() / 2), 30))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j],
                             (TOP_LEFT_X + j * BLOCK_SZ, TOP_LEFT_Y + i * BLOCK_SZ, BLOCK_SZ, BLOCK_SZ))

    # draw grid lines and border
    draw_grid_lines(surface)
    pygame.draw.rect(surface, GRAY, (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5)


def main():
    state = GameState()
    clock = pygame.time.Clock()
    fall_time = 0

    while not state.lost:
        fall_speed = 0.5  # 0.27
        fall_time += clock.get_rawtime()
        clock.tick()

        # PIECE FALLING CODE
        if fall_time / 1000 >= fall_speed:
            fall_time = 0

            # state = state.do_action(GameState.DOWN)

        # EVENTS - PLAYER INPUT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    state = state.do_action(GameState.LEFT)
                elif event.key == pygame.K_RIGHT:
                    state = state.do_action(GameState.RIGHT)
                elif event.key == pygame.K_DOWN:
                    state = state.do_action(GameState.DOWN)
                elif event.key == pygame.K_UP:
                    state = state.do_action(GameState.ROTATE)
                elif event.key == pygame.K_SPACE:
                    state = state.do_action(GameState.HARD_DROP)

        """
        shape_pos = convert_shape_format(state.current)

        # add current piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                state.grid[y][x] = state.current.color
        """

        shape_pos = convert_shape_format(state.current)

        result = state.result()
        score = get_eval_score(result)

        # add current piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                result.grid[y][x] = result.current.color

        draw_window(win, result.grid)
        draw_next_shape(state.next, win)
        draw_eval_score(score, win)
        pygame.display.update()

    draw_text_middle("You Lost", 40, WHITE, win)
    pygame.display.update()
    pygame.time.delay(2000)


def main_menu():
    run = True
    while run:
        win.fill((0, 0, 0))
        draw_text_middle('Press any key to begin.', 60, WHITE, win)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                main()
    pygame.quit()


win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
pygame.display.set_caption('Tetris')

main_menu()  # start game
