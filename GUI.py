import pygame
from playground import run_playground
import numpy as np
from DQN_playground import run_DQN_playground


def main():
    #
    obstacles = [((0, 2), (0, 4), (5, 2), (5, 4)), ((3, 7), (3, 9), (12, 7), (12, 9))]
    rows = 12
    cols = 12
    terminal = np.asarray([rows - 1, cols - 1])
    states, _ = run_DQN_playground(rows, cols, terminal=terminal, obstacles=obstacles)
    print(len(states))

    pygame.init()
    screen = pygame.display.set_mode([1200, 650])
    screen.fill((255, 255, 255))
    pygame.display.set_caption('wow')

    running = True

    for st in states:
        st = f'{st[0]}|{st[1]}'
        pygame.time.delay(200)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        step = 30

        for j in range(rows):
            for i in range(cols):

                s = (step + 10) * j
                rec = pygame.Rect((step + 10) * i, s, step, step)
                k = f'{j}|{i}'

                color = None
                if obstacles is not None:
                    for lu, ru, ld, rd in obstacles:
                        if lu[0] <= j <= rd[0] and lu[1] <= i <= rd[1]:
                            color = (255,0,0)

                if k == st:
                    color = (0, 255, 0)
                elif color is None:
                    color = (0, 0, 255)

                pygame.draw.rect(screen, color, rec, 0)

        pygame.display.update()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()