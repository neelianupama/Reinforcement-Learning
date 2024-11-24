import numpy as np


class GridWorld:


    def __init__(self, tot_row, tot_col):
        self.action_space_size = 4
        self.world_row = tot_row
        self.world_col = tot_col
        #The world is a matrix of size row x col x 2
        #The first layer contains the obstacles
        #The second layer contains the rewards
        #self.world_matrix = np.zeros((tot_row, tot_col, 2))
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size))/ self.action_space_size
        #self.transition_array = np.ones(self.action_space_size) / self.action_space_size
        self.reward_matrix = np.zeros((tot_row, tot_col))
        self.state_matrix = np.zeros((tot_row, tot_col))
        self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]


    #def setTransitionArray(self, transition_array):
        #if(transition_array.shape != self.transition_array):
            #raise ValueError('The shape of the two matrices must be the same.')
        #self.transition_array = transition_array        


    def setTransitionMatrix(self, transition_matrix):
        '''Set the reward matrix.


        The transition matrix here is intended as a matrix which has a line
        for each action and the element of the row are the probabilities to
        executes each action when a command is given. For example:
        [[0.55, 0.25, 0.10, 0.10]
         [0.25, 0.25, 0.25, 0.25]
         [0.30, 0.20, 0.40, 0.10]
         [0.10, 0.20, 0.10, 0.60]]


        This matrix defines the transition rules for all the 4 possible actions.
        The first row corresponds to the probabilities of executing each one of
        the 4 actions when the policy orders to the robot to go UP. In this case
        the transition model says that with a probability of 0.55 the robot will
        go UP, with a probaiblity of 0.25 RIGHT, 0.10 DOWN and 0.10 LEFT.
        '''
        if(transition_matrix.shape != self.transition_matrix.shape):
            raise ValueError('The shape of the two matrices must be the same.')
        self.transition_matrix = transition_matrix


    def setRewardMatrix(self, reward_matrix):
        '''Set the reward matrix.


        '''
        if(reward_matrix.shape != self.reward_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.reward_matrix = reward_matrix


    def setStateMatrix(self, state_matrix):
        '''Set the obstacles in the world.


        The input to the function is a matrix with the
        same size of the world
        -1 for states which are not walkable.
        +1 for terminal states
         0 for all the walkable states (non terminal)
        The following matrix represents the 4x3 world
        used in the series "dissecting reinforcement learning"
        [[0,  0,  0, +1]
         [0, -1,  0, +1]
         [0,  0,  0,  0]]
        '''
        if(state_matrix.shape != self.state_matrix.shape):
            raise ValueError('The shape of the matrix does not match with the shape of the world.')
        self.state_matrix = state_matrix


    def setPosition(self, index_row=None, index_col=None):
        ''' Set the position of the robot in a specific state.


        '''
        if(index_row is None or index_col is None): self.position = [np.random.randint(tot_row), np.random.randint(tot_col)]
        else: self.position = [index_row, index_col]


    def render(self):
        ''' Print the current world in the terminal.


        O represents the robot position
        - respresent empty states.
        # represents obstacles
        * represents terminal states
        '''
        graph = ""
        for row in range(self.world_row):
            row_string = ""
            for col in range(self.world_col):
                if(self.position == [row, col]): row_string += u" \u25CB " # u" \u25CC "
                else:
                    if(self.state_matrix[row, col] == 0): row_string += ' - '
                    elif(self.state_matrix[row, col] == -1): row_string += ' # '
                    elif(self.state_matrix[row, col] == +1): row_string += ' * '
            row_string += '\n'
            graph += row_string
        print (graph)            


    def reset(self, exploring_starts=False):
        ''' Set the position of the robot in the bottom left corner.


        It returns the first observation
        '''
        if exploring_starts:
            while(True):
                row = np.random.randint(0, self.world_row)
                col = np.random.randint(0, self.world_col)
                if(self.state_matrix[row, col] == 0): break
            self.position = [row, col]
        else:
            self.position = [self.world_row-1, 0]
        #reward = self.reward_matrix[self.position[0], self.position[1]]
        return self.position


    def step(self, action):
        ''' One step in the world.


        [observation, reward, done = env.step(action)]
        The robot moves one step in the world based on the action given.
        The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        @return observation the position of the robot after the step
        @return reward the reward associated with the next state
        @return done True if the state is terminal  
        '''
        if(action >= self.action_space_size):
            raise ValueError('The action is not included in the action space.')


        #Based on the current action and the probability derived
        #from the trasition model it chooses a new actio to perform
        action = np.random.choice(4, 1, p=self.transition_matrix[int(action),:])
        #action = self.transition_model(action)


        #Generating a new position based on the current position and action
        if(action == 0): new_position = [self.position[0]-1, self.position[1]]   #UP
        elif(action == 1): new_position = [self.position[0], self.position[1]+1] #RIGHT
        elif(action == 2): new_position = [self.position[0]+1, self.position[1]] #DOWN
        elif(action == 3): new_position = [self.position[0], self.position[1]-1] #LEFT
        else: raise ValueError('The action is not included in the action space.')


        #Check if the new position is a valid position
        #print(self.state_matrix)
        if (new_position[0]>=0 and new_position[0]<self.world_row):
            if(new_position[1]>=0 and new_position[1]<self.world_col):
                if(self.state_matrix[new_position[0], new_position[1]] != -1):
                    self.position = new_position


        reward = self.reward_matrix[self.position[0], self.position[1]]
        #Done is True if the state is a terminal state
        done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward, done


#Gridworldtest.py:
import numpy as np
from gridworld import GridWorld


env = GridWorld(3, 4)


#Define the state matrix
state_matrix = np.zeros((3,4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1


#Define the reward matrix
reward_matrix = np.full((3,4), -0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1


#Define the transition matrix
transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                              [0.1, 0.8, 0.1, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.1, 0.0, 0.1, 0.8]])


#Define the policy matrix
policy_matrix = np.array([[1,  1,  1,  0],
                          [0, -1,  0,  0],
                          [0,  3,  3,  3]])




env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)
env.setTransitionMatrix(transition_matrix)


#Reset the environment
observation = env.reset()
env.render()


for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    observation, reward, done = env.step(action)
    print("")
    print("ACTION: " + str(action))
    print("REWARD: " + str(reward))
    print("DONE: " + str(done))
    env.render()
    if done: break


#Montecarlo_control:
import numpy as np
from gridworld import GridWorld


def print_policy(policy_matrix):
    '''Print the policy using specific symbol.


    * terminal state
    ^ > v < up, right, down, left
    # obstacle
    '''
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(policy_matrix[row,col] == -1): policy_string += " *  "            
            elif(policy_matrix[row,col] == 0): policy_string += " ^  "
            elif(policy_matrix[row,col] == 1): policy_string += " >  "
            elif(policy_matrix[row,col] == 2): policy_string += " v  "          
            elif(policy_matrix[row,col] == 3): policy_string += " <  "
            elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)


def get_return(state_list, gamma):
    '''Get the return for a list of action-state values.


    @return get the Return
    '''
    counter = 0
    return_value = 0
    for visit in state_list:
        # (observation, action, reward ) = visit
        _, _, reward = visit
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


def update_policy(episode_list, policy_matrix, state_action_matrix):
    '''Update a policy making it greedy in respect of the state-action matrix.


    @return the updated policy
    '''
    for visit in episode_list:
        # (observation, action, reward ) = visit
        observation, _, _ = visit
        col = observation[1] + (observation[0]*4)
        if(policy_matrix[observation[0], observation[1]] != -1):      
            policy_matrix[observation[0], observation[1]] = \
                np.argmax(state_action_matrix[:,col])
    return policy_matrix






def main():


    env = GridWorld(3, 4)


    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)


    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)


    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])


    #Random policy
    policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
    policy_matrix[1,1] = np.NaN #NaN for the obstacle at (1,1)
    policy_matrix[0,3] = policy_matrix[1,3] = -1 #No action for the terminal states


    #Set the matrices in the world
    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)


    state_action_matrix = np.random.random_sample((4,12)) # Q
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((4,12), 1.0e-10)
    gamma = 0.999
    tot_epoch = 500000
    print_epoch = 3000


    for epoch in range(tot_epoch):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=True)
        #action = np.random.choice(4, 1)
        #action = policy_matrix[observation[0], observation[1]]
        #episode_list.append((observation, action, reward))
        is_starting = True
        for _ in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]
            #If the episode just started then it is
                #necessary to choose a random action (exploring starts)
            if(is_starting):
                action = np.random.randint(0, 4)
                is_starting = False      
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            #Append the visit in the episode list
            episode_list.append((observation, action, reward))
            observation = new_observation
            if done: break
        #The episode is finished, now estimating the utilities
        counter = 0
        #Checkup to identify if it is the first visit to a state
        checkup_matrix = np.zeros((4,12))
        #This cycle is the implementation of First-Visit MC.
        #For each state stored in the episode list check if it
        #is the rist visit and then estimate the return.
        for visit in episode_list:
            observation, action, reward = visit
            col = int(observation[1] + (observation[0]*4))
            row = int(action)
            if(checkup_matrix[row, col] == 0):
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                state_action_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        #Policy Update
        policy_matrix = update_policy(episode_list,
                                      policy_matrix,
                                      state_action_matrix/running_mean_matrix)
        #Printing
        if(epoch % print_epoch == 0):
            print("")
            print("State-Action matrix after " + str(epoch+1) + " iterations:")
            print(state_action_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epoch+1) + " iterations:")
            print(policy_matrix)
            print_policy(policy_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(state_action_matrix / running_mean_matrix)




if __name__ == "__main__":
    main()
Montecarlo_prediction:
import numpy as np
from gridworld import GridWorld


def get_return(state_list, gamma):
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[1]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value




def main():


    env = GridWorld(3, 4)


    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)


    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print("Reward Matrix:")
    print(reward_matrix)


    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])


    #Define the policy matrix
    #This is the optimal policy for world with reward=-0.04
    policy_matrix = np.array([[1,      1,  1,  -1],
                              [0, np.NaN,  0,  -1],
                              [0,      3,  3,   3]])
    print("Policy Matrix:")
    print(policy_matrix)


    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)


    utility_matrix = np.zeros((3,4))
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((3,4), 1.0e-10)
    gamma = 0.999
    tot_epoch = 50000
    print_epoch = 1000


    for epoch in range(tot_epoch):
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=False)
        for _ in range(1000):
            #Take the action from the action matrix
            action = policy_matrix[observation[0], observation[1]]
            #Move one step in the environment and get obs and reward
            observation, reward, done = env.step(action)
            #Append the visit in the episode list
            episode_list.append((observation, reward))
            if done: break
        #The episode is finished, now estimating the utilities
        counter = 0
        #Checkup to identify if it is the first visit to a state
        checkup_matrix = np.zeros((3,4))
        #This cycle is the implementation of First-Visit MC.
        #For each state stored in the episode list check if it
        #is the rist visit and then estimate the return.
        for visit in episode_list:
            observation = visit[0]
            row = observation[0]
            col = observation[1]
            reward = visit[1]
            if(checkup_matrix[row, col] == 0):
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                utility_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        if(epoch % print_epoch == 0):
            print("")
            print("Utility matrix after " + str(epoch+1) + " iterations:")
            print(utility_matrix / running_mean_matrix)
    #Time to check the utility matrix obtained
    print("Utility matrix after " + str(tot_epoch) + " iterations:")
    print(utility_matrix / running_mean_matrix)






if __name__ == "__main__":
    main()
Rolling_a_dice.py
import numpy as np


#Trowing a dice for N times and evaluating the expectation
dice = np.random.randint(low=1, high=7, size=3)
print("Expectation (3 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=10)
print("Expectation (10 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100)
print("Expectation (100 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=1000)
print("Expectation (1000 times): " + str(np.mean(dice)))
dice = np.random.randint(low=1, high=7, size=100000)
print("Expectation (100000 times): " + str(np.mean(dice)))




