import numpy as np
from utils import *
np.seterr(divide='ignore')


def assert_dist_non_negative(log_p):
    assert np.all(log_p <= 0)


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
    def __init__(self, T, val_X, val_O, prior, transition_mat, emission_mat):
        self.T = T
        self.val_X = val_X
        self.val_O = val_O
        self.log_prior = np.log(prior)
        self.log_transition_mat = np.log(transition_mat)
        self.log_emission_mat = np.log(emission_mat)

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
    ##########     Est. Prob.     ##########
    ########################################
    def gibbs_sampling_posterior(self, obs, M_max=100, M_start=50):
        """
        TODO(Proj2) For each observation obs[i]:
                1. Sample a starting point X1
                2. Sample Xt given the others and given obs[i] for M iterations
                3. Use the last M-M_start samples to estimate the probability of each Xt given obs p(Xt | o=obs[i])
        Note: you can sample for all N observations simultaneously
        :param obs - N observations. shape = (N,T)
        :param M_max: number of iteration to use for Gibbs (int)
        :return: estimated log posterior for Xt. shape = (N, T, |val(X)|)
        """
        N = obs.shape[0]
        val_len = len(self.val_X)
        log_post_Xt = np.zeros((N, self.T, val_len))
        samples = np.zeros((N,M_max, self.T),dtype= int)
        x1 = np.random.choice(self.val_X, size=(N,self.T))
        samples[:, 0, :] = x1
        for n in range(N):
            for m in range(1, M_max):
                samples[n, m, :] = samples[n, m - 1, :]  # Initialize current sample with the previous one
                curr_samples = samples[n, m, :]
                for t in range(self.T):
                    log_prob = np.zeros(val_len)
                    for val in range(val_len):
                        log_prob[val] = self.log_emission_mat[val, obs[n, t]]
                        if t > 0:
                            log_prob[val] += self.log_transition_mat[samples[n, m, t - 1], val]
                        if t < (self.T - 1):
                            log_prob[val] += self.log_transition_mat[val, samples[n, m, t + 1]]
                    log_prob -= logsumexp(log_prob)
                    sample = np.random.choice(self.val_X, p=np.exp(log_prob))
                    samples[n, m, t] = sample
        #for each observation
        samples = samples[:,M_start:M_max,:]
        for n in range(N):
            for t in range(self.T):
                for val in range(val_len):
                    log_post_Xt[n,t,val] = np.sum(samples[n,:,t]==val)
        log_post_Xt = np.log(log_post_Xt/(M_max-M_start))
        assert_dist_sums_to_1(log_post_Xt, axis=2)  # sum_k[ p(Xt = k | o) ] == 1
        return log_post_Xt

    def likelihood_weighting_posterior(self, obs, M_max=100):
        """
        TODO(Proj2) For each observation obs[i]:
                1. Sample M_max samples from the network
                2. Calculate their LW weights
                3. Use LW to estimate the probability of each Xt given obs p(Xt | o=obs[i])
        Note: you can sample and calculate the weights of all N*M_max samples simultaneously

        :param obs - N observations. shape = (N,T)
        :param M_max: number of samples to use for LW (int)
        :return: estimated log posterior for Xt. shape = (N, T, |val(X)|)
        """
        N = obs.shape[0]
        val_len = len(self.val_X)
        samples = np.zeros((N,M_max,self.T),dtype= int)
        weights = np.ones((N, M_max))
        #sampling and weighting
        for n in range(N):
            for m in range(M_max):
                samples[n, m, 0] = np.random.choice(self.val_X, p=np.exp(self.log_prior))

        for n in range(N):
            for m in range(M_max):
                log_weight = self.log_emission_mat[samples[n,m,0], obs[n, 0]]
                for t in range(1, self.T):
                    samples[n, m, t] = np.random.choice(self.val_X, p=np.exp(self.log_transition_mat[samples[n, m, t - 1]]))
                    log_weight += self.log_emission_mat[samples[n, m, t], obs[n, t]]
                weights[n, m] = log_weight
        weights = np.exp(weights - logsumexp(weights, axis=1, keepdims=True))

        #calculating posterior
        log_post_Xt = np.zeros((N, self.T, val_len))
        for t in range(self.T):
            for val in range(val_len):
                ind = (samples[:,:,t] == val)
                log_post_Xt[:,t,val] = np.log(np.sum(weights*ind,axis =1)/np.sum(weights,axis = 1))

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
        if (log_post_Xt is  None):
            log_post_Xt = self.log_posterior_Xt(obs)
        for n in range(N):
            for t in range(self.T):
                X_hat[n][t] = np.argmax(log_post_Xt[n][t])
        return X_hat

