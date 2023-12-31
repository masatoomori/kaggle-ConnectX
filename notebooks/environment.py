import numpy as np
import gymnasium as gym
from kaggle_environments import make, evaluate

# https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/connectx

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_INVALID = -10

RANDOM_SEED = 0


class ConnectFourGym(gym.Env):
    def __init__(self, opponent='random', switch_prob=0.5):
        self.env = make("connectx", debug=True)
        self.switch_prob = switch_prob
        self.agents = [None, opponent]
        self.trainer = self.env.train(self.agents)

        self.config = self.env.configuration
        # PyTorch Conv2d expect 4 dimensional data
        # (nSamples x nChannels x Height x Width)
        self.board_template = (1, self.config.rows, self.config.columns)

        # 報酬のレンジを設定
        self.reward_range = (REWARD_INVALID, REWARD_WIN)

        self.action_space = gym.spaces.Discrete(self.config.columns)
        self.observation_space = gym.spaces.Box(
            # lowとhighの間の値をとる
            low=0,
            high=2,
            shape=self.board_template,
            dtype=int
        )

    def switch_starting_positions(self):
        """
        ゲームの開始位置を交換するためのメソッドです。
        このメソッドを呼び出すことで、エージェント（プレイヤー）の開始位置が反転されます。

        具体的には、このメソッドは以下の処理を行います：
        1. agentsリストの要素を反転させることで、self.agents[0]とself.agents[1]の順序を入れ替えます。
           これにより、ゲームの開始時にプレイヤーの役割が交換されます。
        2. self.trainerを新しいagentsリストに基づいて再設定します。
           env.train()メソッドは、指定されたエージェントの組み合わせでトレーニング用の環境を作成します。したがって、self.trainerを再設定することで、新しい開始位置のプレイヤーの組み合わせに基づいてトレーニング環境が再構築されます。

        このメソッドの役割は、ランダムにゲームの開始位置を変更することで、トレーニングデータの多様性を増やすことです。
        ゲームの開始位置が異なると、エージェントが異なる状況に対して学習することになります。
        その結果、より汎化性の高いモデルが得られる可能性があります。
        """
        self.agents = self.agents[::-1]
        self.trainer = self.env.train(self.agents)

    def reset(self, seed=RANDOM_SEED):
        """
        resetメソッドの戻り値は、強化学習のエージェントが新しいエピソード（新しいゲーム）を始めるときの初期状態として使用されます。
        強化学習では、タスクは通常エピソードと呼ばれる一連のステップで構成され、各エピソードは環境がリセットされて新たに始まります。
        resetメソッドは、環境を初期状態にリセットし、その初期状態を返します。
        この初期状態（resetメソッドの戻り値）は、エージェントが新しいエピソードの最初の行動を選択する際の観測として使用されます。
        エージェントはこの観測を元に、何らかのポリシー（方策）に従って最初のアクションを選択します。
        したがって、resetメソッドの戻り値は、通常は強化学習エージェントのstepメソッドやactメソッドなど、アクションを選択するメソッドの入力として使われます。
        具体的なメソッド名はエージェントの実装に依存します。
        """
        if np.random.random() < self.switch_prob:
            self.switch_starting_positions()

        # ConnectXゲームのケースでは、self.observationは以下のような情報を持つ辞書型のデータになります。
        # 'board': 現在のゲームボードの状態を表す1次元のリストです。
        #          0は空きスペース、1はエージェントのピース、2は相手のピースを表します。
        #          リストは左上から右下へと、行ごとに並んでいます。
        #          例えば、3x3のボードで中央だけエージェントのピースがある場合、'board'は[0, 0, 0, 0, 1, 0, 0, 0, 0]となります。
        # 'mark':  現在のエージェントのマーク（ピース）を表す数字で、1または2のいずれかです。
        #          そのため、self.observation['board']は現在のゲームボードの状態を、self.observation['mark']はエージェントのマーク（ピース）を取得します。
        # このself.observationは、エージェントが環境の状態を理解し、次にどのようなアクションを取るべきかを決定するための重要な情報源となります。
        self.observation = self.trainer.reset()
        self.board = np.array(self.observation['board']).reshape(self.board_template)
        return self.board, {}

    def update_reward(self, prev_reward, done):
        if prev_reward == REWARD_WIN:
            return REWARD_WIN
        elif done:
            return REWARD_LOSE
        else:
            return 1 / (self.config.rows * self.config.columns)

    def step(self, action):
        """
        stepメソッドの戻り値は通常、RL(強化学習)のエージェントが次の行動を決定する際の入力として使用されます。
        stepメソッドは次の5つの要素を戻り値として返します：観測、報酬、終了フラグ、切り捨てフラグ、および追加情報。
        観測：     この新しい観測（ゲームの状態）は、エージェントが次の行動を決定する際の入力として使用されます。
                  エージェントはこの観測を元に、何らかのポリシー（方策）に従って次にどのアクションを選択するかを決定します。
        報酬：     この報酬は、エージェントが取った前の行動の結果として得られます。
                  この報酬は、エージェントが学習する際に使用されます。
                  つまり、エージェントはこの報酬を用いて、取った行動がどれほど良かったのか（ゲームの勝利にどれほど寄与したのか）を評価し、その結果を元に学習を進めます。
        終了フラグ：ゲームが終了したかどうかを示すこのフラグは、エージェントが次の行動を選択する際に考慮されます。
                  ゲームが終了している場合、エージェントは新たな行動を選択することなく、新たなエピソード（ゲーム）を開始します。
        これらの戻り値は、通常は強化学習エージェントのstepメソッドやactメソッドなど、次の行動を選択するメソッドの入力として使われます。
        これらのメソッドはエージェントごとに異なる可能性がありますので、具体的なメソッド名はエージェントの実装に依存します。
        """
        is_valid = (self.observation['board'][int(action)] == 0)

        if is_valid:
            # self.trainer.step(int(action))メソッドは、KaggleのConnectX環境でエージェントが指定した行動を実行し、その結果を取得するために使われます。
            # 引数：          このメソッドは一つの引数、すなわちint(action)を必要とします。
            #                これはエージェントが選択した行動を表します。
            #                この行動は整数で表され、ConnectXゲームにおいては、エージェントがコマを置く列を表します。
            #                この引数は整数として与えられるため、int(action)としている部分は、行動が整数であることを明示的に保証しています。
            # 実行と結果の取得：self.trainer.step(int(action))は、エージェントが選択した行動をゲーム環境に適用します。
            #                具体的には、エージェントが選択した列にコマを落とす行動を実行します。
            # 戻り値：        このメソッドは次の4つの値を返します：
            # 新しい観測：     エージェントが行動を取った結果としての新しいゲーム状態です。
            # 報酬：          エージェントが取った行動の結果として得られる報酬です。
            # 終了フラグ：     ゲームが終了したかどうかを表すブール値です。
            #                例えば、エージェントまたは対戦相手が勝利した場合、または盤面がすべて埋まった場合などにTrueとなります。
            # 情報：          デバッグや診断に有用な追加情報を含む辞書です。
            #                具体的な内容は環境に依存します。
            # これらの値は、self.observation, reward, done, info = self.trainer.step(int(action))の行によってそれぞれの変数に保存され、後続の処理で利用されます。
            self.observation, reward, done, info = self.trainer.step(int(action))
            reward = self.update_reward(reward, done)
        else:
            reward, done, info = REWARD_INVALID, True, {}    # 妥当なアクションでない場合は最大限の罰を与えて終了

        terminated = truncated = done
        self.board = np.array(self.observation['board']).reshape(self.board_template)
        return self.board, reward, terminated, truncated, info
