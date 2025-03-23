from utils import *
np.seterr(divide='ignore')
EPS = 1e-14


def assert_dist_non_negative(log_p):
    assert np.all(log_p <= EPS)


def assert_dist_sums_to_1(log_p, axis, check_idx=None):
    if check_idx is None: assert np.all(np.isclose(np.exp(logsumexp(log_p, axis=axis)), 1))
    else: assert np.all(np.isclose(np.exp(logsumexp(log_p, axis=axis)[check_idx]), 1))


def _log_forward(data_obs, log_p, log_t_mat, log_e_mat, val_X, T):
    """
    F[1, k] = p(X_1 = k, o_1) = p(o_1 | X_1 = k) * p(X_1 = k) = e[k -> o_1] * p[k]
    F[t, k] = p(X_t = k, o_{1:t}) = e[k -> o_t] * sum_l[ F[t-1, l] * tau[l -> k] ]
    shape = N x T x |Val(X)|
    """
    N = data_obs.shape[0]
    log_F = np.zeros((N, T, len(val_X))) - np.inf
    log_F[:, 0] = log_p + log_e_mat[:, data_obs[:, 0]].T
    for t in range(1, T):
        temp = log_F[:, t - 1, None] + log_t_mat.T[None, :]
        log_F[:, t] = logsumexp(temp + log_e_mat[:, data_obs[:, t]].T[:, :, None], axis=2)

    # tests
    assert_dist_non_negative(log_F)
    return log_F

    N = data_obs.shape[0]
    log_F = np.zeros((N, T, len(val_X))) - np.inf
    log_F[:, 0] = log_p + log_e_mat[:, data_obs[:, 0]].T
    for t in range(1, T):
        temp = log_F[:, t - 1, None] + log_t_mat.T[None, :]
        log_F[:, t] = logsumexp(temp + log_e_mat[:, data_obs[:, t]].T[:, :, None], axis=2)

    # tests
    assert_dist_non_negative(log_F)
    return log_F


def _log_backward(data_obs, log_t_mat, log_e_mat, val_X, T):
    """
    B[T, k] = p(empty_set | X_T = k) = 1
    B[t, k] = p(o_{t+1:T} | X_t = k) = sum_s[ tau[k -> s] * B[t+1, s] * e[s -> o_{t+1}]]
    shape = N x T x |Val(X)|
    """
    N = data_obs.shape[0]
    log_B = np.zeros((N, T, len(val_X)))
    log_B[:, -1] = np.log(1)
    for t in np.arange(T - 1, 0, -1):
        temp = log_B[:, t, None] + log_t_mat[None, :]
        log_B[:, t - 1] = logsumexp(temp + log_e_mat[:, data_obs[:, t]].T[:, None, :], axis=2)

    # tests
    assert_dist_non_negative(log_B)
    return log_B


class HMM:
    def __init__(self, T, val_X, val_O, prior=None, transition_mat=None, emission_mat=None):
        self.T = T
        self.val_X = val_X
        self.val_O = val_O
        self.log_prior = np.log(prior) if prior is not None else None
        self.log_transition_mat = np.log(transition_mat) if transition_mat is not None else None
        self.log_emission_mat = np.log(emission_mat) if emission_mat is not None else None

    def get_CPDs(self):
        return {'prior': np.exp(self.log_prior),
                'transition_mat': np.exp(self.log_transition_mat),
                'emission_mat': np.exp(self.log_emission_mat)}

    def print_CPDs(self):
        cpds = self.get_CPDs()
        k = 'prior'
        print(k)
        print(np.array([f'prior({x})={cpds[k][x]:.3f}' for x in self.val_X]))
        k = 'transition_mat'
        print(k)
        print(np.array([[f'tau({xt}->{xtp1})={cpds[k][xt][xtp1]:.3f}' for xt in self.val_X] for xtp1 in self.val_X]).T)
        k = 'emission_mat'
        print(k)
        print(np.array([[f'e({xt}->{ot})={cpds[k][xt][ot]:.3f}' for ot in self.val_O] for xt in self.val_X]))

    ########################################
    ##########      Sampling      ##########
    ########################################
    def sample(self, N=1):
        """
        TODO(Proj1) sample N samples from the HMM.
        Assumes that the HMM CPDs are defined.
        :param N: optional, default=1. Number of samples.
        :return: (hidden, obs) for N samples from the HMM. shape of hidden & obs = (N,hmm.T)
        """
        hidden = np.zeros((N, self.T), dtype=int)
        obs = np.zeros((N, self.T), dtype=int)
        for n in range(N):
            states = np.arange(len(self.log_prior))
            observations = np.arange(len(self.val_O))
            current_state = np.random.choice(states, p=np.exp(self.log_prior))
            current_obs = np.random.choice(
                observations, p=np.exp(self.log_emission_mat[current_state])
            )
            hidden[n][0] = current_state
            obs[n][0] = current_obs
            if self.T == 1:
                break
            for i in range(1, self.T):
                current_state = np.random.choice(
                    states, p=np.exp(self.log_transition_mat[current_state])
                )
                current_obs = np.random.choice(
                    states, p=np.exp(self.log_emission_mat[current_state])
                )
                hidden[n][i] = current_state
                obs[n][i] = current_obs
        return hidden, obs

    ########################################
    ##########     Calc Prob.     ##########
    ########################################
    def log_joint(self, hidden, obs):
        """
        TODO(Proj1) calculate the log joint probability of the hidden, observations sequences for each sample p(x1:T[t],o1:T[i]).
        :param hidden - N hidden sequences. shape = (N,T)
        :param obs - N observations. shape = (N,T)
        :return log-joint probability. shape = (N)
        """
        log_joint_prob = np.zeros(hidden.shape[0], dtype=float)
        for n in range(hidden.shape[0]):
            hidden_sequence = hidden[n]
            obs_sequence = obs[n]
            prior = self.log_prior[hidden_sequence[0]]
            log_prob = (
                    prior + self.log_emission_mat[hidden_sequence[0]][obs_sequence[0]]
            )
            if hidden.shape[1] == 1:
                log_joint_prob[n] = log_prob
                break
            for i in range(1, len(hidden_sequence)):
                log_prob = (
                        log_prob
                        + self.log_transition_mat[hidden_sequence[i]][
                            hidden_sequence[i - 1]
                        ]
                        + self.log_emission_mat[obs_sequence[i]][hidden_sequence[i]]
                )
            log_joint_prob[n] = log_prob
        return log_joint_prob

    def naive_log_likelihood(self, obs):
        """
        Calculate the log likelihood of the observations for each sample p(o1:T[i]) in a naive way (going over all
        possibilities). This will take many resources for T>5.
        :param obs - N observations. shape = (N,T)
        :return log-likelihood. shape = (N)
        """
        assert self.T < 6
        X = set(itertools.permutations(np.array([[x] * self.T for x in self.val_X]).flatten(), self.T))
        p = []
        for x in X:
            p.append(self.log_joint(np.broadcast_to(x, obs.shape), obs))
        return logsumexp(p, axis=0)

    def log_likelihood(self, obs):
        """
        TODO(Proj1) calculate the log likelihood of the observations for each sample p(o1:T[i]).
             Use the supplied forward-algorithm.
        :param obs - N observations. shape = (N,T)
        :return log-likelihood. shape = (N)
        """
        N = len(obs)
        log_F = _log_forward(obs, self.log_prior, self.log_transition_mat, self.log_emission_mat, self.val_X, self.T)
        log_likelihood = np.zeros(obs.shape[0])
        for n in range(N):
            log_f_n = log_F[n][-1]
            log_likelihood[n] = logsumexp(log_f_n)
        return log_likelihood

    def log_prior_Xt(self):
        """
        TODO(Proj1) calculate the point-wise prior p(X_t=x)
        :return point-wise prior. shape = (T, |val(X)|)
        """
        val_len = len(self.val_X)
        log_p_Xt = np.zeros((self.T, val_len))
        for val in range(val_len):
            log_p_Xt[0, val] = self.log_prior[val]

        for t in range(1, self.T):
            for val in range(val_len):
                dynamic_array = np.zeros(val_len)
                for val_t in range(val_len):
                    dynamic_array[val_t] = self.log_transition_mat[val_t, val] + log_p_Xt[t - 1][val_t]
                log_p_Xt[t, val] = logsumexp(dynamic_array)
        # tests
        assert_dist_non_negative(log_p_Xt)   # p(Xt = k) >= 0
        assert_dist_sums_to_1(log_p_Xt, axis=-1)  # sum_k[ p(Xt = k) ] == 1
        return log_p_Xt

    def log_naive_posterior_Xt(self, obs):
        """
        TODO(Proj1) calculate the point-wise posterior pX_t=x | ot=obs[i][t])
        :param obs - N observations. shape = (N,T)
        :return point-wise posterior. shape = (N, T, |val(X)|)
        """
        N = obs.shape[0]
        val_len = len(self.val_X)
        log_posterior_Xt_given_Ot = np.zeros((N, self.T, val_len))
        log_p_Xt = self.log_prior_Xt()
        for n in range(N):
            for t in range(self.T):
                ot_given_xt = self.log_emission_mat[:, obs[n][t]]
                for val in range(val_len):
                    # log(p(xt = k |ot)) = log((p(ot|xt=k)*p(xt = k))/p(ot))
                    #  = log(p(ot|xt = k)) + log(p(xt = k)) - log(p(ot).
                    # log(p(ot)) =  log(sum_k(p(ot|xt = k)*p(xt = k))
                    # = logsumexp([log(p(ot|xt = 1)+log(p(xt = 1))), ... , log(p(ot|xt = n)+log(p(xt = n))])
                    p_x_t = log_p_Xt[t][val]
                    ot_given_xt_val = ot_given_xt[val]
                    xt_given_ot = p_x_t + ot_given_xt_val - logsumexp(ot_given_xt + log_p_Xt[t])
                    log_posterior_Xt_given_Ot[n][t][val] = xt_given_ot
        # tests
        assert_dist_non_negative(log_posterior_Xt_given_Ot)  # p(Xt = k | ot) >= 0
        assert_dist_sums_to_1(log_posterior_Xt_given_Ot, axis=-1)  # sum_k[ p(Xt = k | ot) ] == 1
        return log_posterior_Xt_given_Ot

    def log_posterior_Xt(self, obs):
        """
        TODO(Proj2) Calculate the posterior p(X_t=x | o=obs[i])
        :param obs - N observations. shape = (N,T)
        :return log posterior for Xt. shape = (N, T, |val(X)|)
        """
        N = obs.shape[0]
        val_len = len(self.val_X)
        log_post_Xt = np.zeros((N, self.T, val_len))
        log_forward = _log_forward(obs,self.log_prior,self.log_transition_mat,self.log_emission_mat,self.val_X,self.T)
        log_backward = _log_backward(obs,self.log_transition_mat,self.log_emission_mat,self.val_X,self.T)
        log_likelihood = self.log_likelihood(obs)
        for n in range(N):
            posterior = np.zeros((self.T, val_len))
            for t in range(self.T):
                for val in range(val_len):
                    posterior[t][val] = log_forward[n][t][val] + log_backward[n][t][val]-log_likelihood[n]
            log_post_Xt[n] = posterior
        assert_dist_sums_to_1(log_post_Xt, axis=2)  # sum_k[ p(Xt = k | o) ] == 1
        return log_post_Xt

    ########################################
    ##########     Prediction     ##########
    ########################################
    def naive_predict_by_naive_posterior(self, obs):
        """
        TODO(Proj1) predict a sequence of hidden states for each sample using the point-wise posterior:
            X_hat[i][t] = argmax_x[ p(X_t=x | ot=obs[i][t]) ]
        :param obs - N observations. shape = (N,T)
        :return X_hat - N hidden sequences. shape = (N,T)
        """
        N = obs.shape[0]
        log_naive_posterior_Xt = self.log_naive_posterior_Xt(obs)
        X_hat = np.zeros((N, self.T))
        for n in range(N):
            for t in range(self.T):
                X_hat[n][t] = np.argmax(log_naive_posterior_Xt[n][t])
        return X_hat

    def naive_predict_by_posterior(self, obs, log_post_Xt=None):
        """
        TODO(Proj2) predict a sequence of hidden states for each sample using the posterior:
            X_hat[i][t] = argmax_x[ p(X_t=x | o=obs[i]) ]
        :param obs - N observations. shape = (N,T)
        :param log_post_Xt - optional. Use it to predict the hidden states. If not given, calculate using
                                        the log_posterior_Xt method
        :return X_hat - N hidden sequences. shape = (N,T)
        """
        N = obs.shape[0]
        X_hat = np.zeros((N, self.T))
        if (log_post_Xt is None):
            log_post_Xt = self.log_posterior_Xt(obs)
        for n in range(N):
            for t in range(self.T):
                X_hat[n][t] = np.argmax(log_post_Xt[n][t])
        return X_hat

    ########################################
    ##########      Learning      ##########
    ########################################
    def update_CPDs(self, log_prior, log_transition_mat, log_emission_mat):
        self.log_prior, self.log_transition_mat, self.log_emission_mat = log_prior, log_transition_mat, log_emission_mat


    ##########         MLE        ##########
    def _get_log_SS(self, hidden, obs):
        """
        Calculates the log of the sufficient statistics from the data.

        :param hidden - N hidden sequences. shape = (N,T)
        :param obs - N observations. shape = (N,T)
        :return: log_counts_initial = log M[X1=k] for each state k,
                 log_counts_transition = log sum_t=1,...,T[ M[Xt=k, Xtp1=l] ] for states k,l,
                 log_counts_emission = log sum_t=1,...,T[ M[Xt=k, Ot=s] ] for each state k and observation s
        """
        K, L = len(self.val_X), len(self.val_O)

        # Initial State Counts
        log_counts_initial = np.log(np.bincount(hidden[:, 0], minlength=K))

        # Transition Counts
        transitions = hidden[:, :-1] * K + hidden[:, 1:]
        log_counts_transition = np.log(np.bincount(transitions.flatten(), minlength=K**2).reshape(K, K))

        # Emission Counts
        emissions = hidden * L + obs
        log_counts_emission = np.log(np.bincount(emissions.flatten(), minlength=K * L).reshape(K, L))

        return log_counts_initial, log_counts_transition, log_counts_emission


    def _get_MLE(self, log_counts_initial, log_counts_transition, log_counts_emission):
        """
        Maximizes the parameters based on the log sufficient statistics.

        :param See return value of the _get_log_SS method
        :return the log of the CPDs MLE: prior, transition and emission CPDs
        """
        # p(x1) = #{X1=x1} / N
        log_prior = log_counts_initial - logsumexp(log_counts_initial)

        # p(xt -> xtp1) = p(xtp1 | xt) = p(xtp1 , xt) / p(xt) = #{Xt=xt, Xtp1=xtp1} / #{Xt=xt}  (t=1,...,T-1)
        norm = logsumexp(log_counts_transition, axis=1)
        log_transition_mat = log_counts_transition - norm[:, None]

        # p(xt -> ot) = p(ot | xt) = p(ot , xt) / p(xt) = #{Xt=xt, Ot=ot} / #{Xt=xt}  (t=1,...,T)
        norm = logsumexp(log_counts_emission, axis=1)
        log_emission_mat = log_counts_emission - norm[:, None]

        # tests
        assert_dist_sums_to_1(log_prior, axis=0)
        assert_dist_sums_to_1(log_transition_mat, axis=1)
        assert_dist_sums_to_1(log_emission_mat, axis=1)
        return log_prior, log_transition_mat, log_emission_mat

    def _trainMLE(self, hidden, obs):
        """
        Learns the maximum likelihood estimates for the HMM parameters.

        :param hidden - N hidden sequences. shape = (N,T)
        :param obs - N observations. shape = (N,T)
        :return the log of the CPDs MLE: prior, transition and emission CPDs
        """
        log_counts_initial, log_counts_transition, log_counts_emission = self._get_log_SS(hidden, obs)
        log_prior, log_transition_mat, log_emission_mat = self._get_MLE(
            log_counts_initial, log_counts_transition, log_counts_emission
        )
        return log_prior, log_transition_mat, log_emission_mat


    ##########         EM         ##########
    def _Estep(self, obs, log_prior, log_transition_mat, log_emission_mat):
        """
        TODO:
            Calculate the log of the expected sufficient statistics from the observations.
            Return the log ESS and the log likelihood of the observations under the current model.

        :param obs - N observations. shape = (N,T)
        :param log_prior, log_transition_mat, log_emission_mat - starting point for the CPDs
        :return: log_expected_counts_initial = log \tilde{M}[X1=k] for each state k,
                 log_expected_counts_transition = log sum_t=1,...,T[  \tilde{M}[Xt=k, Xtp1=l] ] for states k,l,
                 log_expected_counts_emission = log sum_t=1,...,T[  \tilde{M}[Xt=k, Ot=s] ] for each state k and observation s
                 where \tilde{M}[y] is the expectation of M[y] w.r.t the posterior.
                 Also returns log_likelihood = log likelihood of the observations (float) = sum over the ll of all samples
        """
        T = obs.shape[1]
        val_x = log_transition_mat.shape[0]
        log_expected_counts_initial = np.zeros(val_x)
        log_expected_counts_transition_t = np.full((T, val_x, val_x), -np.inf)  # ndarray, shape=(#states, #states)
        log_expected_counts_transition = np.full((val_x, val_x), -np.inf)  # ndarray, shape=(#states, #states)
        log_expected_counts_emission = np.full((val_x, val_x), -np.inf)
        # Compute forward and backward probabilities
        log_forward = _log_forward(obs, log_prior, log_transition_mat, log_emission_mat, self.val_X, T)
        log_backward = _log_backward(obs, log_transition_mat, log_emission_mat, self.val_X, T)
        # Compute log likelihood
        log_likelihood = logsumexp(log_forward[:, -1], axis=1)
        for val in range(val_x):
            log_initial_p = logsumexp(log_forward[:, 0, val] + log_backward[:, 0, val] - log_likelihood)
            log_expected_counts_initial[val] = log_initial_p

        for k in range(val_x):
            for l in range(val_x):
                for t in range(T - 1):
                    val = log_forward[:, t, k] + log_transition_mat[k, l] + log_emission_mat[
                        l, obs[:, t + 1]] + log_backward[:, t + 1, l]
                    val -= log_likelihood
                    log_expected_counts_transition_t[t, k, l] = logsumexp(val, axis=0)
                log_expected_counts_transition[k,l] = logsumexp(log_expected_counts_transition_t[:,k,l], axis=0)
        log_expected_counts_transition -= logsumexp(log_expected_counts_transition, axis=1, keepdims=True)

        for k in range(val_x):
            for l in range(val_x):
                log_sum_terms = np.full((obs.shape[0], T), -np.inf)
                for t in range(T):
                    # Create a mask for the current observation l
                    mask = (obs[:, t] == l)
                    if np.any(mask):
                        val = log_forward[mask, t, k] + log_backward[mask, t, k] - log_likelihood[mask]
                        log_sum_terms[mask, t] = val
                log_expected_counts_emission[k, l] = logsumexp(log_sum_terms, axis=(0, 1))
         #normalize
        log_expected_counts_emission -= logsumexp(log_expected_counts_emission, axis=1, keepdims = True)

        return log_expected_counts_initial, log_expected_counts_transition, log_expected_counts_emission, log_likelihood.sum()

    def _Mstep(self, log_expected_counts_initial, log_expected_counts_transition, log_expected_counts_emission):
        """
        TODO: Maximize the parameters based on the log expected sufficient statistics.
        :param See return value of the _Estep method
        :return the log of the CPDs MLE: prior, transition and emission CPDs
        """
        return self._get_MLE(log_expected_counts_initial,log_expected_counts_transition,log_expected_counts_emission)


    def _trainEM(self, obs, start, max_itr=10):
        """
        Trains the Hidden Markov Model using the Expectation-Maximization (EM) algorithm.

        The EM algorithm iteratively updates the parameters of the HMM to maximize the likelihood
        of the observed data. This method runs the EM algorithm until the maximum number of iterations is reached.

        :param obs - ndarray of shape (N, T)
            The observed data sequences. Each row corresponds to an observation sequence of length T.
        :param start - the starting point of the EM - dict('log_prior', 'log_transition_mat', 'log_emission_mat')
        :param max_iter - int, optional, default=10
            The maximum number of iterations to run the EM algorithm.

        :return log_prior, log_transition_mat, log_emission_mat - the train CPDs after running EM for L iterations
        :return traj - the trajectory of the parameters -
                       list of L tuples (log_prior in itr l, log_transition_mat in itr l, log_emission_mat in itr l)
        :return ll_traj - the trajectory of the parameters -
                       list (length L) of the log likelihood of the observations in each iteration l
        """

        log_prior, log_transition_mat, log_emission_mat = start['log_prior'], start['log_transition_mat'], start['log_emission_mat']

        traj = [(log_prior, log_transition_mat, log_emission_mat)]
        ll_traj = []
        for l in tqdm(range(max_itr)):
            log_expected_counts_initial, log_expected_counts_transition, log_expected_counts_emission, ll = self._Estep(
                obs, log_prior, log_transition_mat, log_emission_mat
            )
            log_prior, log_transition_mat, log_emission_mat = self._Mstep(
                log_expected_counts_initial, log_expected_counts_transition, log_expected_counts_emission
            )
            traj.append((log_prior, log_transition_mat, log_emission_mat))
            ll_traj.append(ll)

        return log_prior, log_transition_mat, log_emission_mat, traj, ll_traj
