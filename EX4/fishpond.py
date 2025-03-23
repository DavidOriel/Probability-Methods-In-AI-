######################################################################################################################
# HUJI 67800 - Probabilistic Methods in AI
# Reinforcement Learning Programming Assignment
#     ******** The Fish Pond ********
# Eitan Richardson 2018
######################################################################################################################

import numpy as np
import csv
from matplotlib import pyplot as plt


class FishPond:
    """
    The Fish Pond Environment
    Few (if any) changes are needed in this class.
    Please implement fish agent logic in a separate file (agentfish.py)
    Notes: The word "current" sometimes means "present" and sometimes "flow of water"... don't be confused.
    """
    def __init__(self, pondfile):
        self.__read_pond_file(pondfile)
        self.current_directions = [(-1, 0), (0, 1), (0, -1), (1, 0)] # Left, Up, Down, Right
        self.action_directions = {'l': (-1, 0), 'u': (0, 1), 'd': (0, -1), 'r': (1, 0)}
        self.pond_size = (len(self.currents[0]), len(self.currents))  # (width, height)
        self.__create_transition_matrices()
        self.reset()

    def __create_transition_matrices(self):
        """
        Convert the textual state currents (4 chars) to list of (prob, new_state) tuples
        """
        max_current = 2
        self.transition_lists = {}
        for x in range(self.pond_size[0]):
            for y in range(self.pond_size[1]):
                current_state = (x, y)
                transitions_list = []
                for i in range(4):
                    magnitude = int(self.currents[y][x][i])
                    if magnitude > 0:
                        offset = [magnitude * self.current_directions[i][j] for j in range(2)]
                        new_state = tuple([max(0, min(self.pond_size[j]-1, current_state[j] + offset[j])) for j in range(2)])
                        transitions_list.append(new_state)
                if transitions_list:
                    self.transition_lists[(x, y)] = [(1.0/len(transitions_list), t) for t in transitions_list]
                else:
                    self.transition_lists[(x, y)] = [(1.0, current_state)]

    def __read_pond_file(self, pondfile):
        """
        Read the text file describing the pong currents and start and end location
        """
        self.currents = []
        with open(pondfile, 'r') as infile:
            reader  = csv.reader(infile)
            start_end = [int(v) for v in next(reader)]
            self.start_state = tuple(start_end[:2])
            self.end_state = tuple(start_end[2:])
            for row in reader:
                self.currents.append(row)
        self.currents = self.currents[::-1]

    def reset(self):
        self.current_state = self.start_state
        self.trajectory = [self.start_state]

    def get_action_outcomes(self, state, action):
        """
        Get access to the fishpond transition matrix (without changing the current state)
        :param state: Original state (x,y) tuple
        :param action: Action ('u' / 'd' / 'r' / 'l')
        :return: list of (p(new_state|state, action), new_state) tuples
        """
        temp_state = tuple([max(0, min(self.pond_size[i]-1, state[i] + self.action_directions[action][i]))
                            for i in range(2)])
        return self.transition_lists[temp_state]

    def perform_action(self, action):
        """
        Take an action (affects current_state)
        :param action: one of 'l', 'r', 'u', 'd'
        :return reached_end (boolean)
        """
        t_list = self.get_action_outcomes(self.current_state, action)
        new_state = t_list[np.argmax(np.random.multinomial(1, [t[0] for t in t_list]))][1]
        # print(len(self.trajectory), ':', self.current_state, '--', action ,'-->', new_state)
        self.current_state = new_state
        self.trajectory.append(new_state)
        return tuple(self.current_state) == tuple(self.end_state)

    def plot(self, values=None):
        """
        Plot the fish pond, currents, start, end and current state and optionally calculated calues
        :param values: Optional - np array of size (h, w) with calculated state values
        """
        plt.cla()
        plt.xlim([0, self.pond_size[0]])
        plt.ylim([0, self.pond_size[1]])
        plt.xticks(np.arange(self.pond_size[0]), [])
        for i in range(self.pond_size[0]):
            plt.text(i+0.4, -0.5, str(i))
        plt.yticks(np.arange(self.pond_size[1]), [])
        for i in range(self.pond_size[1]):
            plt.text(-0.5, i+0.4, str(i))

        # Draw the trajectory
        t_x = np.array([t[0] for t in self.trajectory])
        t_y = np.array([t[1] for t in self.trajectory])
        plt.plot(t_x+0.5, t_y+0.5, 'r-o')

        # Draw currents and values
        for x in range(self.pond_size[0]):
            for y in range(self.pond_size[1]):
                if values is not None:
                    plt.text(x, y, '%.1f'%values[y, x])
                c = self.currents[y][x]
                assert len(c)==4
                for i in range(4):
                    if c[i] != '0':
                        head_size = 0.15 if c[i] == '1' else 0.3
                        d = self.current_directions[i]
                        plt.arrow(x+0.5-0.4*d[0], y+0.5-0.4*d[1], (0.8-head_size)*d[0], (0.8-head_size)*d[1],
                                  head_width=head_size, head_length=head_size, overhang=1.0)

        # Draw start and end states
        plt.gcf().gca().add_artist(plt.Circle((self.start_state[0]+0.5, self.start_state[1]+0.5), 0.4, color='r', alpha=0.5))
        plt.gcf().gca().add_artist(plt.Circle((self.end_state[0]+0.5, self.end_state[1]+0.5), 0.4, color='g', alpha=0.5))
        plt.gcf().gca().add_artist(plt.Circle((self.current_state[0]+0.5, self.current_state[1]+0.5), 0.25, color='b', alpha=0.5))
        plt.grid(True)
        plt.pause(0.2)


