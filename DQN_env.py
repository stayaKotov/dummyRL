import numpy as np


class DQNEnvironement:
    def __init__(
            self,
            init_state,
            terminal_state,
            step_reward=-1,
            final_reward=100,
            bound_x=None,
            bound_y=None,
            obstacles=None
    ):
        self.terminal_state = terminal_state
        self.init_state = init_state
        self.step_reward = step_reward
        self.final_reward = final_reward
        self.obstacles = obstacles
        self.bound_x = bound_x
        self.bound_y = bound_y
        self.tuple = self.reset()
        self.y, self.x = self.tuple[0], self.tuple[1]

    def reset(self, use_random=False):
        if not use_random:
            tmp = self.init_state
        else:
            tmp = np.asaray([np.random.choice(15), np.random.choice(2)])
        self.tuple = np.hstack([tmp, self.terminal_state])
        self.y, self.x = self.tuple[0], self.tuple[1]
        return self.tuple

    def check_boundaries(self, y, x):
        out_bound = False
        if 0 <= x < self.bound_x and 0 <= y < self.bound_y:
            return y, x, out_bound
        else:
            if x < 0:
                x = 0
            elif x > self.bound_x:
                out_bound = True
                x = self.bound_x

            if y < 0:
                y = 0
            elif y > self.bound_y:
                out_bound = True
                y = self.bound_y
        return y, x, out_bound

    def check_obstacles(self, y, x, cur_y, cur_x):
        in_obstacle = False
        if self.obstacles is None:
            return y, x, in_obstacle

        for lu, ru, ld, rd in self.obstacles:
            if lu[1] <= x <= rd[1] and lu[0] <= y <= rd[0]:
                x = cur_x
                y = cur_y
                in_obstacle = True
        return y, x, in_obstacle

    def step(self, action):
        self.y, self.x = self.tuple[0], self.tuple[1]
        tmp_y = self.y
        tmp_x = self.x
        if action == 0:  # up
            tmp_y = self.y - 1
        elif action == 1:  # 'down'
            tmp_y = self.y + 1
        elif action == 2:  # 'left'
            tmp_x = self.x - 1
        elif action == 3:  # 'right'
            tmp_x = self.x + 1

        tmp_y, tmp_x, is_out_bound = self.check_boundaries(tmp_y, tmp_x, )
        tmp_y, tmp_x, is_in_obst = self.check_obstacles(tmp_y, tmp_x, self.y, self.x)

        self.y = tmp_y
        self.x = tmp_x

        reward, is_done = self.__get_reward()
        if is_out_bound:
            reward = -20
        if is_in_obst:
            reward = -20

        self.tuple = np.asarray([self.y, self.x])
        self.tuple = np.hstack([self.tuple, self.terminal_state])

        return self.tuple, reward, is_done

    def __get_reward(self):
        finish_y, finish_x = self.terminal_state[0], self.terminal_state[1]
        if (self.y, self.x) == (finish_y, finish_x):
            return self.final_reward, True
        else:
            return self.step_reward, False
