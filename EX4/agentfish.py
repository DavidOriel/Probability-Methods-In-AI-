######################################################################################################################
# HUJI 67800 - Probabilistic Methods in AI
# Reinforcement Learning Programming Assignment
#     ******** The Fish Pond ********
# Eitan Richardson 2018
######################################################################################################################
import time

import numpy as np
from fishpond import FishPond
from matplotlib import pyplot as plt
import random
# TODO: Implement your reward, policy and policy evaluation and improvement functions here


class AgentFish:
    def __init__(self, pond: FishPond, alpha, gamma, epsilon ):
        self.pond = pond
        self.target = self.pond.end_state
        self.alpha = alpha
        self.gamma = gamma
        self.actions = ["l", "r", "u", "d"]
        self.states = [(x, y) for x in range(pond.pond_size[0]) for y in range(pond.pond_size[1])]  # A 3x3 grid
        self.V = {state: 0 for state in self.states}
        self.q_table = {(state, action):0 for state in self.states for action in self.actions}
        self.policy= {state: None for state in self.states}
        self.epsilon = epsilon
############################################# MDP methods###########################3
    def plot_episode(self,episode):
        for step in episode:
            action = step[1]
            reached_end = self.pond.perform_action(action)
            self.pond.plot()
            plt.show()
        self.pond.reset()

    #value iteration
    def reward(self, state, action):
        if self.get_next_state(state,action) == self.pond.end_state:
            return 10
        return -0.1

    def get_next_state(self,state,action):
        outcomes = self.pond.get_action_outcomes(state, action)
        probabilities, states = zip(*outcomes)
        next_state = random.choices(states, probabilities)[0]
        return next_state

    def optimal_policy_plot(self,max_iterations):
        for j in range(max_iterations):
            action = self.policy[self.pond.current_state]
            reached_end = self.pond.perform_action(action)
            self.pond.plot()
            plt.show()
            if reached_end == True:
                break
        self.pond.reset()

# Example policy:
    def the_right_policy(self):
        return 'r'

    def q0_sample_policy(self):
    # Sample code for running a policy and plotting the trajectory
        for i in range(30):
            action = self.the_right_policy()
            reached_end = self.pond.perform_action(action)
            self.pond.plot()
            if reached_end:
                break
        print('Done')
        plt.savefig('Q0_'+pond_name+'.png')
        plt.show()

####################################################Q1#################################3
    def greedy_policy_one_step(self,state,target):
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        move_options = []
        if dx < 0:
            move_options.append("l")
        elif dx > 0:
            move_options.append("r")
        if dy < 0:
            move_options.append("d")
        elif dy > 0:
            move_options.append("u")
        if len(move_options) > 1:
            return random.choice(move_options)
        elif move_options:
            return move_options[0]

    def generate_greedy_episodes(self,num_episodes,epsilon):
        state = self.pond.current_state
        target = self.pond.end_state
        distance = np.abs(target[0] - state[0]) + np.abs(target[1] - state[1])
        episodes = []
        for i in range(num_episodes):
            episode = []
            for j in range(distance * 3):
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(self.actions)
                else:
                    action = self.greedy_policy_one_step(state,target)
                reward = self.reward(state, action)
                episode.append((state,action,reward))
                if state == target:
                    break
                state = self.get_next_state(state,action)
            state = self.pond.current_state
            episodes.append(episode)
        return episodes

#########################################################q2###############################
    def q2_value_iteration(self,iterations = 50):
        for i in range(iterations):
            for state in self.states:
                self.V[state] = max(self.reward(state, action) + self.gamma*sum(p* self.V[state_prime] for p,state_prime in self.pond.get_action_outcomes(state,action)) for action in self.actions)
        return self.V

    def q2_find_best_policy(self):
        for state in self.states:
            max_action = max(self.actions, key = lambda a: self.reward(state, a) + self.gamma*sum(p* self.V[state_prime] for p,state_prime in self.pond.get_action_outcomes(state,a)))
            self.policy[state] =max_action
        return self.policy


#########################################################q3########################################3
    def update_policy_from_Q(self):
        for state in self.states:
            a = self.choose_max_action(state)
            self.policy[state] = a

    def choose_max_action(self, state):
        q_values = [self.q_table[(state, a)] for a in self.actions]
        max_q_value = max(q_values)
        max_actions = [a for a, q in zip(self.actions, q_values) if q == max_q_value]
        return random.choice(max_actions)

    def q3_generate_greedy_episode(self):
        while True:
            state = self.pond.current_state
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(self.actions)
            else:
                action = self.greedy_policy_one_step(state, self.target)
            reached_end = self.pond.perform_action(action)
            reward = self.reward(state, action)
            next_state = self.pond.current_state
            best_action = self.choose_max_action(next_state)
            a = self.q_table[(next_state, best_action)]
            q_target = reward + self.gamma * self.q_table[(next_state, best_action)]
            self.q_table[(state, action)] += self.alpha * (q_target - self.q_table[(state, action)])
            if reached_end:
                break
        self.pond.reset()

    def q3_generate_action_based_episode(self):
        while True:
            state = self.pond.current_state
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(self.actions)
            else:
                action = self.choose_max_action(state)
            reached_end = self.pond.perform_action(action)
            reward = self.reward(state, action)
            next_state = self.pond.current_state
            best_action = self.choose_max_action(next_state)
            a = self.q_table[(next_state, best_action)]
            q_target = reward + self.gamma * self.q_table[(next_state, best_action)]
            self.q_table[(state, action)] += self.alpha * (q_target - self.q_table[(state, action)])
            if reached_end:
                break
        self.pond.reset()

    def q3_q_learning_offline(self, max_episodes):
        # Q -learning algo
        for episode in range(max_episodes):
            q_table_prev = self.q_table.copy()
            self.q3_generate_greedy_episode()
            q_change = np.max(np.abs(np.array(list(self.q_table.values())) - np.array(list(q_table_prev.values()))))
            if q_change < 1e-3:  # Convergence criteria based on Q-value change
                convergence_episodes = episode
                print(episode)
                break
        return self.q_table

    def q3_q_learning_online(self, max_episodes):
        first_episode = True
        for episode in range(max_episodes):
            q_table_prev = self.q_table.copy()
            if first_episode == True:
                self.q3_generate_greedy_episode()
                first_episode = False
            else:
                self.q3_generate_action_based_episode()
            q_change = np.max(np.abs(np.array(list(self.q_table.values())) - np.array(list(q_table_prev.values()))))

            if q_change < 1e-3:  # Convergence criteria based on Q-value change
                convergence_episodes = episode
                print(episode)
                break
        return self.q_table

#####################################################################################################


if __name__ == "__main__":
    pond_name = ('pond1')
    my_pond = FishPond(pond_name+'.txt')
    agent_fish = AgentFish(my_pond, 0.1, 0.99, 0.2
                           )
    #q1
    episodes = agent_fish.generate_greedy_episodes(10,epsilon=0)
    agent_fish.plot_episode(random.choice(episodes))

    #q2
    agent_fish.q2_value_iteration()
    agent_fish.q2_find_best_policy()
    agent_fish.optimal_policy_plot(50)

    #q3
    start_time = time.time()
    agent_fish.q3_q_learning_online(400)
    agent_fish.update_policy_from_Q()
    print("duration for online learning:", time.time() - start_time)
    agent_fish.optimal_policy_plot(50)

    agent_fish2 = AgentFish(my_pond, 0.1, 0.99, 0.2)

    start_time = time.time()
    agent_fish2.q3_q_learning_offline(400)
    agent_fish2.update_policy_from_Q()
    print("duration for offline learning:", time.time() - start_time)
    agent_fish2.optimal_policy_plot(50)


    #q4
    pond_name = ('pond4')
    my_pond = FishPond(pond_name+'.txt')
    agent_fish = AgentFish(my_pond, 0.1, 0.99, 0.2
                           )
    # 4.1
    episodes = agent_fish.generate_greedy_episodes(10,epsilon=0)
    agent_fish.plot_episode(random.choice(episodes))

    #4.2
    agent_fish.q2_value_iteration()
    agent_fish.q2_find_best_policy()
    agent_fish.optimal_policy_plot(50)

    #4.3
    start_time = time.time()
    agent_fish.q3_q_learning_online(400)
    agent_fish.update_policy_from_Q()
    print("duration for online learning:", time.time() -start_time )
    agent_fish.optimal_policy_plot(50)

    agent_fish2 = AgentFish(my_pond, 0.1, 0.99, 0.2)
    start_time = time.time()
    agent_fish2.q3_q_learning_offline(400)
    agent_fish2.update_policy_from_Q()
    print("duration for offline learning:", time.time() -start_time )
    agent_fish2.optimal_policy_plot(50)


