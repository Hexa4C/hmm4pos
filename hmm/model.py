import numpy as np
from tqdm import tqdm


class hmm_model():
    """
    Hidden Markov Model for supervised learning
    """
    def __init__(self, n_state, n_observe):
        super().__init__()
        self.transition = np.zeros((n_state, n_state))
        self.observe = np.zeros((n_state, n_observe))
        self.pi = np.zeros((n_state))
        self.n_state = n_state
        self.n_observe = n_observe


    def init_probmat(self, series):
        """
        @Args:

        series: list of series, each serie is a tuple of series of observations and states
        """
        self.transition = np.zeros((self.n_state, self.n_state))
        self.observe = np.zeros((self.n_state, self.n_observe))
        self.pi = np.zeros((self.n_state))
        n_series = len(series)
        trans_base = np.zeros((self.n_state))
        obs_base = np.zeros((self.n_state))
        for cop in tqdm(series):
            sent = cop[0]
            attr = cop[1]
            self.pi[attr[0]] += 1
            for i in range(len(sent) - 1):
                self.transition[attr[i]][attr[i + 1]] += 1
                trans_base[attr[i]] += 1
            for i in range(len(sent)):
                self.observe[attr[i]][sent[i]] += 1
                obs_base[attr[i]] += 1
        self.transition = self.transition / trans_base[:, None]
        self.observe = self.observe / obs_base[:, None]
        self.pi = self.pi / n_series


    def forward(self, observations):
        alphas = np.zeros((1, self.n_state))
        states = list(range(len(self.n_state)))
        T = len(observations)
        # initialize
        for s in states:
            pi_s = self.pi[s]
            alphas[0][s] = pi_s * self.observe[s][observations[0]]
        # recursive
        for t in range(1, T):
            prev_alphas = alphas
            alphas = np.zeros((1, self.n_state))
            for s in states:
                alpha_each = np.sum(np.dot(prev_alphas, self.transition)) * self.observe[s][observations[t]]
                alphas[0][s] = alpha_each
        # terminate
        prob = np.sum(alphas)
        return prob


    def viterbi(self, observations):
        T = len(observations)
        states = list(range(self.n_state))
        delta = np.zeros((T, self.n_state))
        phi = np.zeros((T, self.n_state), dtype=int)
        for s in states:
            delta[0][s] = self.pi[s] * self.observe[s][observations[0]]
            phi[0][s] = 0
        for t in range(1, T):
            for s in states:
                max_arg = 0
                max_prob = 0.0
                for j in states:
                    p = delta[t - 1][j] * self.transition[j][s]
                    if p > max_prob:
                        max_prob = p
                        max_arg = j
                delta[t][s] = max_prob * self.observe[s][observations[t]]
                phi[t][s] = max_arg
        max_final_prob = 0.0
        max_final_state = 0
        for s in states:
            if delta[T - 1][s] > max_final_prob:
                max_final_prob = delta[T - 1][s]
                max_final_state = s
        best_path = [max_final_state]
        for t in range(T - 1, 0, -1):
            prev_state = phi[t][best_path[-1]]
            best_path.append(prev_state)
        best_path = list(reversed(best_path))
        return best_path