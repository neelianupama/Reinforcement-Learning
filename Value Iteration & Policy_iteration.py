import numpy as np


def return_state_utility(v, T, u, reward, gamma):
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)


def main():


    #Change as you want
    state = 8 #it corresponds to (1,1) in the robot world
    #Assuming that the discount factor is equal to 1.0
    gamma = 1.0


    #Starting state vector
    #The agent starts from (1, 1)
    v = np.zeros(12)
    v[state] = 1.0


    #Transition matrix loaded from file
    #(It is too big to write here)
    T = np.load("T.npy")


    #Utility vector
    u = np.array([[0.812, 0.868, 0.918,   1.0,
                   0.762,   0.0, 0.660,  -1.0,
                   0.705, 0.655, 0.611, 0.388]])


    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])


    #Use the Beelman equation to find the utility of state (1,1)
    utility = return_state_utility(v, T, u, r[state], gamma)
    print("Utility of the state: " + str(utility))


if __name__ == "__main__":
    main()



#Policy_iteration.py:
import numpy as np


def return_policy_evaluation(p, u, r, T, gamma):
    for s in range(12):
        if not np.isnan(p[s]):
            v = np.zeros((1,12))
            v[0,s] = 1.0
            action = int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return u


def return_policy_evaluation_linalg(p, r, T, gamma):
    u = np.zeros(12)
    for s in range(12):
        if not np.isnan(p[s]):
            action = int(p[s])
            u[s] = np.linalg.solve(np.identity(12) - gamma*T[:,:,action], r)[s]
    return u


def return_expected_action(u, T, v):
    
    actions_array = np.zeros(4)
    for action in range(4):
         #Expected utility of doing a in state s, according to T and u.
         actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return np.argmax(actions_array)


def print_policy(p, shape):
    counter = 0
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[counter] == -1): policy_string += " *  "            
            elif(p[counter] == 0): policy_string += " ^  "
            elif(p[counter] == 1): policy_string += " <  "
            elif(p[counter] == 2): policy_string += " v  "          
            elif(p[counter] == 3): policy_string += " >  "
            elif(np.isnan(p[counter])): policy_string += " #  "
            counter += 1
        policy_string += '\n'
    print(policy_string)




def main_iterative():
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")


    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1


    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])


    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])


    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        u1 = u.copy()
        u = return_policy_evaluation(p, u, r, T, gamma)
        #Stopping criteria
        delta = np.absolute(u - u1).max()
        if delta < epsilon * (1 - gamma) / gamma: break
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(u, T, v)        
                if a != p[s]: p[s] = a
        print_policy(p, shape=(3,4))


    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")




def main_linalg():
    """Finding the solution using a linear algebra approach


    """
    gamma = 0.999
    iteration = 0
    T = np.load("T.npy")


    #Generate the first policy randomly
    # Nan=Nothing, -1=Terminal, 0=Up, 1=Left, 2=Down, 3=Right
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5] = np.NaN
    p[3] = p[7] = -1


    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])


    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])


    while True:
        iteration += 1
        epsilon = 0.0001
        #1- Policy evaluation
        #u1 = u.copy()
        u = return_policy_evaluation_linalg(p, r, T, gamma)
        #Stopping criteria
        #delta = np.absolute(u - u1).max()
        #if (delta < epsilon * (1 - gamma) / gamma) or iteration > 100: break
        unchanged = True
        for s in range(12):
            if not np.isnan(p[s]) and not p[s]==-1:
                v = np.zeros((1,12))
                v[0,s] = 1.0
                #2- Policy improvement
                a = return_expected_action(u, T, v)        
                if a != p[s]:
                    p[s] = a
                    unchanged = False
        print_policy(p, shape=(3,4))


        if unchanged: break


    print("=================== FINAL RESULT ==================")
    print("Iterations: " + str(iteration))
    #print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Epsilon: " + str(epsilon))
    print("===================================================")
    print(u[0:4])
    print(u[4:8])
    print(u[8:12])
    print("===================================================")
    print_policy(p, shape=(3,4))
    print("===================================================")


def main():


    main_iterative()
    #main_linalg()




if __name__ == "__main__":
    main()



#Transition_matrix_generation:

import numpy as np




def return_transition(row, col, action, tot_row, tot_col):


    if(row > tot_row-1 or col > tot_col-1):
        print("ERROR: the index is out of range...")
        return None


    extended_world = np.zeros((tot_row+2, tot_col+2))


    #If the state is on the grey-obstacle it returns all zeros
    if(row == 1 and col == 1): return extended_world[1:4, 1:5]
    #If the process is on the final reward state it returns zeros
    if(row == 0 and col == 3): return extended_world[1:4, 1:5]
    #If the process is on the final punishment state then returns zeros
    if(row == 1 and col == 3): return extended_world[1:4, 1:5]


    if(action=="up"):
            col += 1
            row += 1
            extended_world[row-1, col] = 0.8
            extended_world[row, col+1] = 0.1  
            extended_world[row, col-1] = 0.1          
    elif(action=="down"):
            col += 1
            row += 1
            extended_world[row+1, col] = 0.8
            extended_world[row, col+1] = 0.1  
            extended_world[row, col-1] = 0.1
    elif(action=="left"):
            col += 1
            row += 1
            extended_world[row-1, col] = 0.1
            extended_world[row+1, col] = 0.1  
            extended_world[row, col-1] = 0.8
    elif(action=="right"):
            col += 1
            row += 1
            extended_world[row-1, col] = 0.1
            extended_world[row+1, col] = 0.1  
            extended_world[row, col+1] = 0.8


    #Reset the obstacle
    if(extended_world[2, 2] != 0): extended_world[row, col] += extended_world[2, 2]
    extended_world[2, 2] = 0.0


    #Control bouncing
    for row in range(0, 5):  
            if(extended_world[row, 0] != 0): extended_world[row, 1] += extended_world[row, 0]
            if(extended_world[row, 5] != 0): extended_world[row, 4] += extended_world[row, 5]
    for col in range(0, 6):
            if(extended_world[0, col] != 0): extended_world[1, col] += extended_world[0, col]
            if(extended_world[4, col] != 0): extended_world[3, col] += extended_world[4, col]


    return extended_world[1:4, 1:5]




def main():
    #T = return_transition(row=2, col=0, action="up")
    #T = return_transition(row=0, col=1, action="down")
    #T = return_transition(row=1, col=3, action="left")
    #T = return_transition(row=2, col=1, action="up")
    #print(T)


    T = np.zeros((12, 12, 4))
    counter = 0
    for row in range(0, 3):
        for col in range(0, 4):
            line = return_transition(row, col, action="up", tot_row=3, tot_col=4)
            T[counter, : , 0] = line.flatten()
            line = return_transition(row, col, action="left", tot_row=3, tot_col=4)
            T[counter, : , 1] = line.flatten()
            line = return_transition(row, col, action="down", tot_row=3, tot_col=4)
            T[counter, : , 2] = line.flatten()
            line = return_transition(row, col, action="right", tot_row=3, tot_col=4)
            T[counter, : , 3] = line.flatten()


            counter += 1


    #print(T[:,:,3])
    u = np.array([[0.0, 0.0, 0.0 ,0.0,
                   0.0, 0.0, 0.0 ,1.0,
                   0.0, 0.0, 0.0 ,0.0]])


    #u = np.zeros((1, 12))


    print(np.dot(u, T[:,:,2]))


    print("Saving T in 'T.npy' ...")
    np.save("T", T)
    print("Done!")


if __name__ == "__main__":
    main()



















#Value_Iteration:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def return_state_utility(v, T, u, reward, gamma):
    """Return the utility of a single state.


    This is an implementation of the Bellman equation.
    """
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:,:,action])))
    return reward + gamma * np.max(action_array)


def generate_graph(utility_list):
    """Given a list of utility arrays (one for each iteration)
       it generates a matplotlib graph and save it as 'output.jpg'


    """
    name_list = ('(1,3)', '(2,3)', '(3,3)', '+1', '(1,2)', '#', '(3,2)', '-1', '(1,1)', '(2,1)', '(3,1)', '(4,1)')
    color_list = ('cyan', 'teal', 'blue', 'green', 'magenta', 'black', 'yellow', 'red', 'brown', 'pink', 'gray', 'sienna')
    counter = 0
    index_vector = np.arange(len(utility_list))
    for state in range(12):
        state_list = list()
        for utility_array in utility_list:
             state_list.append(utility_array[state])
        plt.plot(index_vector, state_list, color=color_list[state], label=name_list[state])  
        counter += 1
    #Adjust the legend and the axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.4), ncol=3, fancybox=True, shadow=True)
    plt.ylim((-1.1, +1.1))
    plt.xlim((1, len(utility_list)-1))
    plt.ylabel('Utility', fontsize=15)
    plt.xlabel('Iterations', fontsize=15)
    plt.savefig("./output.jpg", dpi=500)


def main():
    #Change as you want
    tot_states = 12
    gamma = 0.999 #Discount factor
    iteration = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value


    #List containing the data for each iteation
    graph_list = list()


    #Transition matrix loaded from file (It is too big to write here)
    T = np.load("T.npy")


    #Reward vector
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])    


    #Utility vectors
    u = np.array([0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0,
                   0.0, 0.0, 0.0,  0.0])
    u1 = np.array([0.0, 0.0, 0.0,  0.0,
                    0.0, 0.0, 0.0,  0.0,
                    0.0, 0.0, 0.0,  0.0])


    while True:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(tot_states):
            reward = r[s]
            v = np.zeros((1,tot_states))
            v[0,s] = 1.0
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s]))
        #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
                print("=================== FINAL RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                print(u[0:4])
                print(u[4:8])
                print(u[8:12])
                print("===================================================")
                break


    generate_graph(graph_list)


if __name__ == "__main__":
    main()
