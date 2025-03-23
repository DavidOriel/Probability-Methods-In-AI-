import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.special import logsumexp
import itertools
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import json
CLR = list(mcolors.TABLEAU_COLORS)


def load_data(file_name, data_path='data/'):
    return pd.read_csv(data_path + file_name + '.csv', index_col=0).values


def accuracy(true, pred):
    return np.mean(true == pred)


def add_labels(ax, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if xlim is not None: ax.set_ylim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)


def add_legend(labels, colors, ax=plt, fontsize=10):
    # Sets patches for legend in ax, matches label to color by order of lists
    patches = []
    for l, c in zip(labels, colors[:len(labels)]):
        patches.append(mpatches.Patch(color=c, label=l))
    ax.legend(handles=patches, fontsize=fontsize)
    return patches


def plot_barplot(a, b, labels, xlabel=None, ylabel=None, title=None, figname=None, ylim=(0,1.1)):
    fig, ax = plt.subplots(1, 1)
    temp = pd.DataFrame([a, b], columns=np.arange(len(a)), index=labels).T
    temp.plot.bar(rot=0, width=.5, ax=ax)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f'plots/{figname}.png')
    plt.show()


def plot_with_errorbars(data, x_ticks, ax, label=None, title=None, xlabel=None, ylabel=None,
                        ylims=None, c='black'):
    mean_data = data.mean(axis=0)
    std_data = data.std(axis=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(x_ticks, mean_data, label=label, c=c, marker='o' if data.shape[0] > 1 else None)
    if data.shape[0] > 1: ax.errorbar(x_ticks, mean_data, std_data, c=c, capsize=3, capthick=1)
    if ylims is not None: ax.set_ylim(ylims)
    return ax


def plot_posterior_vs_M(est_posteriors, exact_posteriors, algo_name, M_arr, t=5, x=1):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        axs.flat[i].plot(M_arr, [exact_posteriors[i, t, x]] * len(M_arr), label='Exact', c='tab:orange', marker='o')
        plot_with_errorbars(est_posteriors[:, :, i, t, x], x_ticks=M_arr, ax=axs.flat[i],
                            label=algo_name, title=f'obs[{i}]',
                            xlabel='M', ylabel=f'posterior - p(X{t}={x} | obs[{i}])',
                            ylims=(-0.1, 1.1))
    axs.flat[0].legend()
    fig.suptitle(f'{algo_name} vs Exact posterior for t={t}')
    fig.tight_layout()
    plt.savefig(f'plots/{algo_name}_est_vs_exact_posterior.png')
    plt.show()


def plot_1D_CPD_vs_arr(true_CPD_mat, pred_CPD_mat, labels, arr, ax, ylims=(-0.1, 1.1), clr=CLR):
    i = 0
    for x in range(true_CPD_mat.shape[0]):
        ax.plot(arr, [true_CPD_mat[x]] * len(arr), c='black', marker='o' if len(arr) < 10 else None)
        plot_with_errorbars(pred_CPD_mat[:, :, x].T, x_ticks=arr, ax=ax, ylims=ylims, c=clr[i], label=labels[i])
        ax.legend(fontsize=8)
        i += 1


def plot_2D_CPD_vs_arr(true_CPD_mat, pred_CPD_mat, labels, arr, ax, ylims=(-0.1, 1.1), clr=CLR):
    i = 0
    for x in range(true_CPD_mat.shape[0]):
        for y in range(true_CPD_mat.shape[1]):
            ax.plot(arr, [true_CPD_mat[x, y]] * len(arr), c='black', marker='o' if len(arr) < 10 else None)
            plot_with_errorbars(pred_CPD_mat[:, :, x, y].T, x_ticks=arr, ax=ax, ylims=ylims, c=clr[i], label=labels[i])
            ax.legend(fontsize=8)
            i += 1


def plot_pred_CPDs_vs_M(true_CPDs, pred_CPDs, M_arr, val_X, val_O, title, figname):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    # plot prior
    plot_1D_CPD_vs_arr(true_CPD_mat=true_CPDs['prior'], pred_CPD_mat=np.array(pred_CPDs['prior']),
                       labels=[f'X1={xt}' for xt in val_X], arr=M_arr, ax=axs[0])
    add_labels(axs[0], title='Prior prob.', xlabel='#samples', ylabel='prior prob.')

    # plot transition
    plot_2D_CPD_vs_arr(true_CPD_mat=true_CPDs['transition_mat'], pred_CPD_mat=np.array(pred_CPDs['transition_mat']),
                       labels=[f'Xt={xt}, Xt+1={xtp1}' for xt in val_X for xtp1 in val_X], arr=M_arr, ax=axs[1])
    add_labels(axs[1], title='Transition prob.', xlabel='#samples', ylabel='transition prob.')

    # plot emission
    plot_2D_CPD_vs_arr(true_CPD_mat=true_CPDs['emission_mat'], pred_CPD_mat=np.array(pred_CPDs['emission_mat']),
                       labels=[f'Xt={xt}, Ot={ot}' for xt in val_X for ot in val_O], arr=M_arr, ax=axs[2])
    add_labels(axs[2], title='Emission prob.', xlabel='#samples', ylabel='emission prob.')

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(f'plots/{figname}.png')
    plt.show()


def plot_log_likelihood_vs_itr(itrs, lls, title, figname):
    fig, ax = plt.subplots(1, 1)
    N = lls.shape[0]
    plot_with_errorbars(lls.T / N, x_ticks=itrs, ax=ax, c='black')
    add_labels(ax, xlabel='itr', ylabel='log-likelihood / #samples',
        title='log likelihood curve')
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(f'plots/{figname}.png')
    plt.show()


def plot_trajectory(true_CPDs, traj, val_X, val_O, title, figname):
    traj_arr = np.arange(len(traj))
    traj_prior = np.array([np.exp(traj[s][0]) for s in traj_arr])
    traj_transition = np.array([np.exp(traj[s][1]) for s in traj_arr])
    traj_emission = np.array([np.exp(traj[s][2]) for s in traj_arr])

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    # plot prior
    plot_1D_CPD_vs_arr(true_CPD_mat=true_CPDs['prior'], pred_CPD_mat=traj_prior[:, None],
                       labels=[f'p(X1={xt})={traj_prior[-1, xt]:.2f}' for xt in val_X],
                       arr=traj_arr, ax=axs[0])
    add_labels(axs[0], title='Prior prob.', xlabel='itr', ylabel='prior prob.')

    # plot transition
    plot_2D_CPD_vs_arr(
        true_CPD_mat=true_CPDs['transition_mat'], pred_CPD_mat=traj_transition[:, None],
        labels=[f'p(Xt+1={xtp1} | Xt={xt})={traj_transition[-1, xt, xtp1]:.2f}' for xt in val_X for xtp1 in val_X],
        arr=traj_arr, ax=axs[1]
    )
    add_labels(axs[1], title='Transition prob.', xlabel='itr', ylabel='transition prob.')

    # plot emission
    plot_2D_CPD_vs_arr(
        true_CPD_mat=true_CPDs['emission_mat'], pred_CPD_mat=traj_emission[:, None],
        labels=[f'p(Ot={ot} | Xt={xt})={traj_emission[-1, xt, ot]:.2f}' for xt in val_X for ot in val_O],
        arr=traj_arr, ax=axs[2]
    )
    add_labels(axs[2], title='Emission prob.', xlabel='itr', ylabel='emission prob.')

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(f'plots/{figname}.png')
    plt.show()
