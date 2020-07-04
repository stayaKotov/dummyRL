from env import Environement
from agent import Agent
import tqdm
import numpy as np


class Playground():
    def __init__(
            self,
            episodes=500,
            limit=100,
            epsilon=0.1,
            gamma=0.3,
            lr=0.3,
            env: Environement=None
    ):
        self.episodes = episodes
        self.limit = limit
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

    def play(self):
        print('init....')
        for ep in tqdm.tqdm(range(self.episodes)):
            self.env.reset()
            # текущее состояние агента
            state = self.env.agent.get_state()
            # print(f'state = {state}')
            for lim in range(self.limit):
                # буквами
                action = self.env.egreedy_policy(self.epsilon)
                self.env.step(action)
                # новое состояние агента
                new_state = self.env.agent.get_state()
                # награда и флаг "закончилась ли игра?"
                reward, is_done = self.env.get_reward()
                td_target = reward + self.gamma * np.max(self.env.q_values[new_state])
                td_error = td_target - self.env.q_values[state][self.env.agent.action_2_index[action]]
                self.env.q_values[state][self.env.agent.action_2_index[action]] += self.lr * td_error

                state = new_state

                if is_done:
                    break

    def create_episode(self, is_random=False):
        states = []
        self.env.reset(is_random)
        state = self.env.agent.get_state()
        states.append(state)
        for kkk in range(100):
            ix = np.argmax(self.env.q_values[state])
            action = self.env.agent.action_names[ix]
            self.env.step(action)
            new_state = self.env.agent.get_state()
            _, is_done = self.env.get_reward()
            state = new_state

            states.append(state)
            if is_done:
                break

        return states


if __name__ == "__main__":
    rows = 10
    cols = 10
    agent = Agent(
        x=None, y=None,
        bound_x=cols-1,
        bound_y=rows-1
    )
    env = Environement(
        agent,
        init_state=(0, 0),
        terminal_state=f'{rows-1}|{cols-1}',
        step_reward=-1,
        final_reward=100,
        field_row=rows,
        field_cols=cols,
        obstacles=[((3, 4), (3, 8), (8, 4), (8, 8)),]
    )
    pg = Playground(
        episodes=1000,
        limit=50,
        epsilon=0.3,
        gamma=0.3,
        lr=0.1,
        env=env
    )
    pg.play()

    states = pg.create_episode()
    print(len(states))
    print(states)

    for r in range(rows):
        st = []
        for c in range(cols):
            a = np.round(np.max(pg.env.q_values[f'{r}|{c}']),3)
            ix = np.argmax(pg.env.q_values[f'{r}|{c}'])
            pg.env.agent.action_names[ix]
            st.append((a, pg.env.agent.action_names[ix]))

        print(st)
        print()

    # print(states)
