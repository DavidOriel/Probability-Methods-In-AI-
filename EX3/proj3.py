from HMM_proj3 import *
import os


def get_hmm():
    # define HMMs
    val_X = np.arange(2)
    val_O = np.arange(2)
    hmm_params = {
        'prior': np.array([1, 0]),
        'transition_mat': np.array([[0.6, 0.4],  # p(X_{t+1}=x | X_t=0)
                                    [0, 1]]),  # p(X_{t+1}=x | X_t=1)
        'emission_mat': np.array([[0.9, 0.1],  # p(O_t=o | X_t=0)
                                  [0.01, 0.99]])  # p(O_t=o | X_t=1)
    }
    T = 300

    hmm = HMM(T=T, val_X=val_X, val_O=val_O, prior=hmm_params['prior'],
              transition_mat=hmm_params['transition_mat'], emission_mat=hmm_params['emission_mat'])
    print(f'Load HMM. CPDs:')
    hmm.print_CPDs()
    print()
    print()
    return hmm


########################################
##########        MLE         ##########
########################################
def Q3(true_hmm, hidden_data, obs_data, M_arr, n_repeats=10):
    """
    For each M in M_arr, samples a dataset of M datapoints (x1:T, o1:T) from the given data,
    and uses this dataset and the method train_MLE to get the MLE estimation on M random samples.
a    Repeats the process n_repeats times for each value of M.
    Plots the estimation of each parameter versus M with confident intervals (mu,sigma),
    and adds to the plot the true CPD from which the data was sampled (true_hmm).
    """
    data_idx = np.arange(obs_data.shape[0])
    true_CPDs = true_hmm.get_CPDs()

    pred_hmm = HMM(T=true_hmm.T, val_X=true_hmm.val_X, val_O=true_hmm.val_O)
    pred_CPDs = {'prior': [[] for M in M_arr], 'transition_mat': [[] for M in M_arr],
                 'emission_mat': [[] for M in M_arr]}
    for M_, M in enumerate(M_arr):
        for k in range(n_repeats):
            idx = np.random.choice(data_idx, size=M, replace=False)
            hidden, obs = hidden_data[idx], obs_data[idx]
            log_prior, log_transition_mat, log_emission_mat = pred_hmm._trainMLE(hidden, obs)

            pred_CPDs['prior'][M_].append(np.exp(log_prior))
            pred_CPDs['transition_mat'][M_].append(np.exp(log_transition_mat))
            pred_CPDs['emission_mat'][M_].append(np.exp(log_emission_mat))

    plot_pred_CPDs_vs_M(
        true_CPDs, pred_CPDs, M_arr, pred_hmm.val_X, pred_hmm.val_O,
        title=f'MLE predicted parameters', figname=f'MLE_pred_CPDs_vs_M'
    )


def Q4(true_hmm, hidden_data, obs_data, M):
    """
    For a specific M= #samples value, prints the CPDs that results in calculating the MLE over the first M datapoints
    in the given data.

    TODO: Choose the minimal value of M that show convergence in the previous question (defined in the main function)
    """
    pred_hmm = HMM(T=true_hmm.T, val_X=true_hmm.val_X, val_O=true_hmm.val_O)
    hidden = hidden_data[:M]
    obs = obs_data[:M]
    log_prior, log_transition_mat, log_emission_mat = pred_hmm._trainMLE(hidden, obs)
    pred_hmm.update_CPDs(log_prior, log_transition_mat, log_emission_mat)
    print('Pred CPDs:')
    pred_hmm.print_CPDs()


########################################
##########         EM         ##########
########################################

def load_start(start_num):
    with open(f"data/start{start_num}.json", "r") as f:
        data_loaded = json.load(f)
        start = {key: np.array(value) for key, value in data_loaded.items()}
    return start


def run_EM(true_hmm, obs, start, L=100):
    """
    Runs EM with a specific start parameters for L iterations, prints the resulted CPDs and return them.
    """
    pred_hmm = HMM(T=true_hmm.T, val_X=true_hmm.val_X, val_O=true_hmm.val_O)

    log_prior, log_transition_mat, log_emission_mat, traj, ll_traj = pred_hmm._trainEM(
        obs, start=start, max_itr=L
    )

    pred_hmm.update_CPDs(log_prior, log_transition_mat, log_emission_mat)
    print('Pred CPDs:')
    pred_hmm.print_CPDs()
    return log_prior, log_transition_mat, log_emission_mat, traj, ll_traj


def Q89(true_hmm, obs_data, M, L=100):
    """
    Uses the observations from P3_data_obs.csv (this time without the hidden values) and take the first M datapoints.
    Samples three different starting point, and run the train_EM method.
    Plots the log-likelihood curve and the parameters curve of the training process for each starting point.

    TODO: Implement the _Estep and _Mstep methods in the HMM class.
    """

    start1 = load_start(1)
    start2 = load_start(2)
    start3 = load_start(3)

    for s_, start in enumerate([start1, start2, start3]):
        log_prior, log_transition_mat, log_emission_mat, traj, ll_traj = run_EM(true_hmm, obs_data[:M], start,
                                                                                L=L)
        plot_log_likelihood_vs_itr(
            np.arange(1, len(ll_traj)),
            np.array(ll_traj[1:])[:, None],
            title=f'EM with start{s_ + 1}, #samples={M}, itr > 1',
            figname=f'EM_log_likelihood_vs_itr_start{s_ + 1}_M={M}_itr more then 1'
        )

        plot_log_likelihood_vs_itr(
            np.arange(len(ll_traj)),
            np.array(ll_traj)[:, None],
            title=f'EM with start{s_ + 1}, #samples={M}',
            figname=f'EM_log_likelihood_vs_itr_start{s_ + 1}_M={M}'
        )

        plot_trajectory(
            true_hmm.get_CPDs(), traj, true_hmm.val_X, true_hmm.val_O,
            title=f'EM with start{s_ + 1}, #samples={M}', figname=f'EM_params_trajectory_start{s_ + 1}_M={M}'
        )


if __name__ == '__main__':
    if not os.path.exists('plots/'):
        os.mkdir('plots/')

    # Load HMM
    hmm1 = get_hmm()

    # Load data
    hidden = load_data('P3_hidden_data')[:, :hmm1.T]
    obs = load_data('P3_obs_data')[:, :hmm1.T]

    # Q3
    print('''
        ########################################
        ##########         Q3         ##########
        ########################################
    ''')
    M_arr = [10, 30, 50, 100, 200, 300]
    Q3(hmm1, hidden, obs, M_arr, n_repeats=10)

    # Q4
    print('''
        ########################################
        ##########         Q4         ##########
        ########################################
    ''')
    chosen_M = 50  # TODO
    Q4(hmm1, hidden, obs, M=chosen_M)

    print('''
        ########################################
        ##########         EM         ##########
        ########################################
    ''')
    # Q8, Q9
    print('''
        ########################################
        ##########        Q8-9        ##########
        ########################################
    ''')
    Q89(hmm1, obs, M=400, L=40)
