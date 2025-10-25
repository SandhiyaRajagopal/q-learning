# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given Reinforcement Learning environment using Q-Learning and comparing the state values with the First Visit Monte Carlo method.

## PROBLEM STATEMENT
For the given frozen lake environment, find the optimal policy applying the Q-Learning algorithm and compare the value functions obtained with that of First Visit Monte Carlo method. Plot graphs to analyse the difference visually.

## Q LEARNING ALGORITHM
# Step 1:
Store the number of states and actions in a variable, initialize arrays to store policy and action value function for each episode. Initialize an array to store the action value function.
# Step 2: 
Define function to choose action based on epsilon value which decides if exploration or exploitation is chosen.
# Step 3:
Create multiple learning rates and epsilon values.
# Step 4: 
Run loop for each episode, compute the action value function but in Q-Learning the maximum action value function is chosen instead of choosing the next state and next action's value. 
# Step 5:
Return the computed action value function and policy. Plot graph and compare with Monte Carlo results.
## Q LEARNING FUNCTION
### Name: SANDHIYA R
### Register Number: 212223240146

```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
      state, done=env.reset(), False
      while not done:
        action=select_action(state, Q, epsilons[e])
        next_state, reward, done, _ = env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```




## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.
<img width="896" height="413" alt="image" src="https://github.com/user-attachments/assets/eea1c54a-5ded-493d-b19e-f68f71f9548d" />
<img width="1112" height="702" alt="image" src="https://github.com/user-attachments/assets/dbf06ca0-d2de-4303-8d13-f7beacef92fe" />
<img width="997" height="158" alt="image" src="https://github.com/user-attachments/assets/96462de5-b1f4-4343-b10f-e37156f99654" />


Include plot comparing the state value functions of Monte Carlo method and Qlearning.
<img width="1080" height="488" alt="image" src="https://github.com/user-attachments/assets/8f5bf5b7-40fe-4055-acef-ef226e2de1f1" />
<img width="1106" height="487" alt="image" src="https://github.com/user-attachments/assets/79dd60fe-638c-45f0-ac63-91afe2162ec9" />


## RESULT:

Therefore, python program to find optimal policy using Q-Learning is developed and state value function obtained is compared with first visit monte carlo.
