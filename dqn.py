import tensorflow as tf
import numpy as np


class DQN_Agent(tf.keras.Model):
    def __init__(self, state, hidden_units, actions_n):
        super(DQN_Agent, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state,))
        self.hiidens = []
        for units in hidden_units:
            layer = tf.keras.layers.Dense(units, activation='relu', kernel_initializer='GlorotNormal')
            # bn = tf.keras.layers.BatchNormalization()
            # do = tf.keras.layers.Dropout(0.2)
            self.hiidens.append(layer)

        self.output_layer = tf.keras.layers.Dense(actions_n, activation='linear', kernel_initializer='RandomNormal')

    # @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hiidens:
            z = layer(z)
            # z = bn(z)
            # z = do(z)
        return self.output_layer(z)


class DQN:
    def __init__(
            self,
            state,
            hidden_units,
            actions_n,
            gamma,
            lr,
            max_experiences,
            min_experiences,
            batch_size
    ):
        self.state = state
        self.hidden_units = hidden_units
        self.actions_n = actions_n
        self.gamma = gamma
        self.lr = lr
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size

        self.model = DQN_Agent(state, hidden_units, actions_n)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    def predict(self, inputs):
        vals = np.atleast_2d((inputs/12).astype('float32'))
        return self.model(vals)

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0

        ids         = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states      = np.asarray(self.experience['s'])[ids]
        actions     = np.asarray(self.experience['a'])[ids]
        next_states = np.asarray(self.experience['s2'])[ids]
        rewards     = np.asarray(self.experience['r'])[ids]
        dones      = np.asarray(self.experience['done'])[ids]

        q_value_next = np.max(TargetNet.predict(next_states), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * q_value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.actions_n),
                axis=1
            )
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return 'random', np.random.choice(self.actions_n)
        else:
            preds = self.predict(np.atleast_2d(states))[0]
            return preds, np.argmax(preds)

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for k in self.experience.keys():
                self.experience[k].pop(0)
        for k, v in exp.items():
            self.experience[k].append(v)

    def copy_weights(self, TrainNet):
        var1 = self.model.trainable_variables
        var2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(var1, var2):
            v1.assign(v2.numpy())




