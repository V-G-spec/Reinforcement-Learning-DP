import numpy as np
from gridworld_ref import GridworldEnv
import time


def policy_eval(policy, env, discount_factor = .9 , theta = 1e-8):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
                If policy[1] == [0.1, 0, 0.9, 0], then it goes up with prob. 0.1 or goes down otherwise.
        env (GridworldEnv) : OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        discount_factor (float): Gamma discount factor.
        theta (float): We stop evaluation once our value function change is less than theta for all states.

    Returns:
        V (numpy list) : Vector of length env.nS representing the value function.
    """
    V = np.zeros(env.nS)
    delta = 1
    while delta>=theta:
        delta = 0
        for s in range(env.nS):
            v = 0 #Value
            for a, a_prob in enumerate(policy[s]): #Iterate over actions
                for  prob, s_dash, reward, _ in env.P[s][a]: #Iterate over next states for each action
                    v += a_prob * prob * (reward + discount_factor * V[s_dash]) #Get expected value
            delta = max(delta, np.abs(v - V[s])) #Max change in value function
            V[s] = v
        
    return V

def policy_iter(env, policy_eval_fn = policy_eval, discount_factor = .9):
    """
    Policy Improvement Algorithm.
    Iteratively evaluate the policy and update it.
    Iteration terminiates if updated policy achieves optimal.

    Args:
        Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns -> (policy, V):
        policy (2d numpy list): a matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
        V (numpy list): V is the value function for the optimal policy.
    """
    
    
    
    # start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_stable = False
    
    while (policy_stable==False): #Because stable = optimal
        
        V = policy_eval(policy, env, discount_factor)
        
        policy_stable = True
        
        for s in range(env.nS):
            
            curr_a = np.argmax(policy[s])
            
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, s_dash, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[s_dash])
            

            best_a = np.argmax(action_values)
            
            # Greedy update
            if curr_a != best_a:
                policy_stable = False
            
            policy[s] = np.eye(env.nA)[best_a]
        


    return policy, V

def value_iteration(env, theta=0.0001, discount_factor=.9):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    

    V = np.zeros(env.nS)
    delta = 1
    
    while delta>=theta:
        
        delta = 0
        
        for s in range(env.nS):
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, s_dash, reward, _ in env.P[s][a]:
                    A[a] += prob * (reward + discount_factor * V[s_dash])
            
            best_value = np.max(A) # Need value at this step and not the action
            # delta across all states so far
            delta = max(delta, np.abs(best_value - V[s]))
            V[s] = best_value    
            
    
    
    policy = np.zeros([env.nS, env.nA]) # deterministic policy using best value. Here we use argmax
    for s in range(env.nS):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, s_dash, reward, _ in env.P[s][a]:
                A[a] += prob * (reward + discount_factor * V[s_dash])
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    
    return policy, V

def main():
    print('--------Policy evaluation--------')
    env = GridworldEnv()
    uniform_policy = np.ones([env.nS, env.nA])/env.nA
    v = policy_eval(uniform_policy, env)
    
    v = v.reshape(env.shape)
    print(v)

    print('---------------------------------')

    print('--------Policy iteration---------')
    start = time.process_time()
    policy, v = policy_iter(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")
    print("\nTime taken by Policy Iteration:", time.process_time() - start)
    print('---------------------------------')

    print('--------Value iteration---------')
    start = time.process_time()
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")
    print("\nTime taken by Value Iteration:", time.process_time() - start)
    
    print('---------------------------------')


if __name__ == '__main__':
    main()
