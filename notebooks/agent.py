import numpy as np
import tensorflow as tf


class DeepModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='sigmoid', kernel_initializer='RandomNormal'))
            self.output_layer = tf.keras.layers.Dense(
                num_actions, activation='linear', kernel_initializer='RandomNormal')

    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = DeepModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}  # 経験を保存するための辞書
        self.max_experiences = max_experiences  # 経験の最大数
        self.min_experiences = min_experiences  # 経験の最小数

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            # 経験が十分溜まっていない場合は何もしない
            return 0

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)  # ミニバッチのサンプルをランダムに抽出
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)  # 次の状態で得られる価値の最大値

        # 教師データ。次の状態が終了状態の場合は即時報酬のみを考慮し、終了状態でない場合は次の状態での即時報酬と未来の報酬（割引されたアクション価値）を考慮
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))    # 損失は実際の価値と予測された価値の間の平均二乗誤差

        variables = self.model.trainable_variables  # 学習可能な変数のリスト
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))  # 勾配法で重みを更新

    def get_actions(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0])
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].numpy()
            for i in range(self.num_actions):
                if state.board[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for (key, value) in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for (v1, v2) in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        ref_model = tf.keras.Sequential()

        ref_model.add(self.model.input_layer)
        for layer in self.model.hidden_layers:
            ref_model.add(layer)
        ref_model.add(self.model.output_layer)

        ref_model.load_weights(path)

    def preprocess(self, state):
        result = state.board[:]
        result.append(state.mark)
        return result
