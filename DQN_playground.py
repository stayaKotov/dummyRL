from DQN_env import DQNEnvironement
from dqn import DQN
import tqdm
import numpy as np


class DQN_Playground():
    def __init__(
            self,
            episodes=500,
            limit=100,
            epsilon=0.1,
            gamma=0.3,
            lr=1e-2,
            copy_step=25,
            # train_net: DQN = None,
            # target_net: DQN = None,
            num_states=4,
            env: DQNEnvironement = None,
            num_actions=4,
            hidden_units=[20, 20],
            max_experiences=100,
            min_experiences=20,
            batch_size=32,

    ):
        self.episodes = episodes
        self.limit = limit
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.train_net = None#train_net
        self.target_net = None#target_net

        self.copy_step = copy_step
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.isdone = False

    def play_game(self):
        rewards = 0
        iter = 0
        losses = []
        n = self.limit
        # начальное состояние агента
        state = self.env.reset()
        while True and n >= 0:
            preds, action = self.train_net.get_action(state, self.epsilon)
            # новое состояние агента, награда и флаг "закончилась ли игра?"
            prev_state = state
            state, reward, is_done = self.env.step(action)

            if is_done:# and self.isdone is False:
                print('DONE', self.limit - n)
                # self.isdone = True

            exp = {'s': prev_state, 'a': action, 'r': reward, 's2': state, 'done': is_done}
            self.train_net.add_experience(exp)
            loss = self.train_net.train(self.target_net)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            iter += 1
            if iter % self.copy_step == 0:
                self.target_net.copy_weights(self.train_net)

            if is_done:# and self.isdone is False:
                # print('DONE')
                # self.isdone = True
                break

            n -= 1
        return rewards, np.mean(losses)

    def get_episode(self):
        states = []
        actions = []
        n = self.limit
        # начальное состояние агента
        state = self.env.reset()
        states.append(state)
        while True and n >= 0:
            pred, action = self.train_net.get_action(state, 0)
            # print(action)
            state, _, is_done = self.env.step(action)
            states.append(state)
            actions.append(action)

            if is_done:
                print('done', self.limit - n)
                break

            n -= 1
        return states, actions

    def learning(self):

        self.train_net = DQN(
            state=self.num_states,
            hidden_units=self.hidden_units,
            actions_n=self.num_actions,
            gamma=self.gamma,
            lr=self.lr,
            max_experiences=self.max_experiences,
            min_experiences=self.min_experiences,
            batch_size=self.batch_size,
        )
        self.target_net = DQN(
            state=self.num_states,
            hidden_units=self.hidden_units,
            actions_n=self.num_actions,
            gamma=self.gamma,
            lr=self.lr,
            max_experiences=self.max_experiences,
            min_experiences=self.min_experiences,
            batch_size=self.batch_size,
        )
        total_rewards = np.empty(self.episodes)
        epsilon = 0.99
        decay = 0.9999
        min_epsilon = 0.1
        print('init..')
        for n in tqdm.tqdm(range(self.episodes)):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, losses = self.play_game()
            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
            # with summary_writer.as_default():
            #     tf.summary.scalar('episode reward', total_reward, step=n)
            #     tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            #     tf.summary.scalar('average loss)', losses, step=n)
            if n % 10 == 0 and n > 0:
                pass
                # print()
                # print(self.check_q_values())
                # print()
                # print(
                #     "episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):",
                #       avg_rewards, "episode loss: ", losses
                # )
        # print("avg reward for last 100 episodes:", avg_rewards)
        # env.close()

    def check_q_values(self, terminal):
        fields = np.zeros(shape=(terminal[0]+1, terminal[1]+1), dtype=object)
        act_2_name = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        for i in range(terminal[0]+1):
            for j in range(terminal[1]+1):
                state = np.asarray([i, j, terminal[0], terminal[1]])
                pred, action = self.train_net.get_action(state, 0)
                action = act_2_name[action]
                q = np.round(np.max(pred),2)
                fields[i, j] = "{0}".format(action)
        return fields


if __name__ == "__main__":
    rows = 10
    cols = 10
    terminal = np.asarray([rows-1, cols-1])

    env = DQNEnvironement(
        init_state=np.asarray([0, 0]),
        terminal_state=terminal,
        step_reward=-1,
        final_reward=100,
        bound_x=cols-1,
        bound_y=rows-1,
        obstacles=None
    )
    pg = DQN_Playground(
        episodes=30,
        limit=150,
        epsilon=0.3,
        gamma=0.8,
        lr=1e-2,
        copy_step=25,
        num_states=4,
        env=env,
        num_actions=4,
        hidden_units=[30, 30],
        max_experiences=300,
        min_experiences=100,
        batch_size=32,
    )
    pg.learning()

    print(pg.check_q_values(terminal))

    print(pg.get_episode())
