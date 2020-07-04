import pygame
from env import Environement
from agent import Agent
from playground import Playground
import numpy as np
import utils

from DQN_env import DQNEnvironement
from DQN_playground import  DQN_Playground


def main():
    # # [((0, 2), (0, 4), (5, 2), (5, 4)), ((3, 6), (3, 8), (10, 6), (10, 8))] #
    #
    obstacles = [((0, 2), (0, 4), (5, 2), (5, 4)), ((3, 6), (3, 9), (14, 6), (14, 9))] # utils.create_obstucles(rows, cols, 1)
    #
    # agent = Agent(x=None, y=None, bound_x=cols-1, bound_y=rows-1)
    # env = Environement(
    #     agent,
    #     init_state=(0, 0),
    #     terminal_state=f'{rows-1}|{cols-1}',
    #     step_reward=-1,
    #     final_reward=100,
    #     field_row=rows,
    #     field_cols=cols,
    #     obstacles=obstacles
    # )
    # pg = Playground(episodes=2000, limit=100, epsilon=0.4, gamma=0.6, lr=0.1, env=env)
    #
    # pg.play()
    #
    # states = pg.create_episode()

    rows = 14
    cols = 12

    terminal = np.asarray([rows - 1, cols - 1])

    env = DQNEnvironement(
        init_state=np.asarray([0, 0]),
        terminal_state=terminal,
        step_reward=-1,
        final_reward=100,
        bound_x=cols-1,
        bound_y=rows-1,
        obstacles=obstacles
    )
    pg = DQN_Playground(
        episodes=500,
        limit=300,
        epsilon=0.5,
        gamma=0.85,
        lr=1e-2,
        copy_step=25,
        num_states=4,
        env=env,
        num_actions=4,
        hidden_units=[30, 30, 30, 30],
        max_experiences=700,
        min_experiences=100,
        batch_size=64,
    )
    pg.learning()

    print(pg.check_q_values(terminal))
    states, _ = pg.get_episode()
    print(states[:5])
    states = [st[:2] for st in states]

    pygame.init()
    screen = pygame.display.set_mode([1200, 650])
    screen.fill((255, 255, 255))
    pygame.display.set_caption('wow')

    running = True
    print(len(states))

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