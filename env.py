import numpy as np

class Environement():
    def __init__(
            self,
            agent,
            init_state,
            terminal_state,
            step_reward=-1,
            final_reward=100,
            field_row=5,
            field_cols=20,
            obstacles=None
    ):
        self.terminal_state = terminal_state
        self.agent = agent
        self.init_state = init_state
        self.step_reward = step_reward
        self.final_reward = final_reward
        self.obstacles = obstacles
        # self.field = np.zeros(field_row, field_cols)
        self.q_values = {
            f'{row}|{col}': [0 for act in self.agent.get_actions_names()]
            for row in np.arange(field_row) for col in np.arange(field_cols)
        }

    def reset(self, use_random=False):
        if not use_random:
            self.agent.y, self.agent.x = self.init_state # np.random.choice(9), 0
        else:
            self.agent.y, self.agent.x = np.random.choice(15), np.random.choice(2)
        self.agent.update_state()

    def step(self, action):
        self.agent.action(action, self.obstacles)

    def get_reward(self):
        if self.agent.state == self.terminal_state:
            return self.final_reward, True
        else:
            return self.step_reward, False

    def egreedy_policy(self, epsilon=0.1):
        acts = self.agent.get_actions_names()
        if np.random.random() < epsilon:
            a = acts[np.random.choice((len(acts)))]
            # print(a)
            return a
        else:
            a = np.argmax(self.q_values[self.agent.state])
            return acts[a]
