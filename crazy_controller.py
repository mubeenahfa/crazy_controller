import gymnasium as gym
import sinergym  
from datetime import datetime, timedelta
import numpy as np
from numpy.random import default_rng

# --- this has been imported to define reward ---
from sinergym.utils.rewards import BaseReward
from typing import Dict, Any, Tuple, List 




#this is a very dumb reward function written below to give you an idea of the structure of a reward function
class DumbReward(BaseReward):
    """
    Crazy controller needs a crazy reward system so it rewards high energy consumption
    and high |pmv| values.
    """
    def __init__(self, energy_var="total_electricity_HVAC", pmv_var="pmv",
                 alpha=1.0, beta=1.0, **kwargs):
       
        super().__init__()
        self.energy_var = energy_var
        self.pmv_var = pmv_var
        self.alpha = alpha
        self.beta = beta

    def __call__(self, obs_dict: Dict[str, Any]):
        energy = obs_dict.get(self.energy_var, 0.0)
        pmv = obs_dict.get(self.pmv_var, 0.0)

        # Rewarding "bad" behavior: maximize energy + maximize |pmv|
        reward = self.alpha * energy + self.beta * abs(pmv)

        # Log terms
        terms = {
            "energy": energy,
            "pmv_abs": abs(pmv),
            "reward": reward,
        }

        # Sinergym's BaseReward expects to return a tuple (reward, terms) so make sure you do the same
        return reward, terms




class CrazyController:
    """
    such a crazy controller right? reads all the observations only to
    generate a random action. Hopefully you will do better in your project.
    """
    def __init__(self, env):
        self.env = env
        self.num_actions = 40  # if you dont know why this is 40, read project documentation again.

    def act(self, observation: List[Any]) -> int:
        try:
            seed_value = int(np.sum(np.abs(observation)) * 1000)
        except Exception:
            seed_value = 0 
        rng = default_rng(seed=seed_value)
        action = rng.integers(0, self.num_actions)        
        return int(action)



# --- Main script execution ---
if __name__ == "__main__":
    
    env_name = 'Eplus-A403mediumfanger-hot-discrete' # use the same environment name unless you specifically wish to test another env
    
    print(f"--- Creating environment '{env_name}'... ---")
    print(f"--- Using Reward: DumbReward ---")


    env = gym.make(env_name,
                   reward=DumbReward)
    
    print("--- Environment created successfully! ---")
    #print("--- Environment's Observation Variables ---")
    #print(env.get_wrapper_attr('observation_variables'))
    #print("--- ACTION SPACE ---")
    #print("Action Space Type:", env.action_space)
    #print("Actuator Names (the 'key' for the action array):", list(env.spec.kwargs['actuators'].keys()))
    action_mapping_function = env.get_wrapper_attr('action_mapping')

   # print("\n--- ACTION MAPPING (what each action number means) ---")
#loop below shows you action numbers and what they map to

    for action_number in range(env.action_space.n):
        try:
            
            real_action_values = action_mapping_function(action_number)
            print(f"Action {action_number}: {real_action_values}")
        except IndexError:
            #observe that this will allow us to see how many actions are actually defined by the mapper
            print(f"Action {action_number}: [ERROR - This action is not defined in the mapping!]")
            
    print("-----------------------")

 
    print("--- Resetting environment... ---")
    observation, info = env.reset()

    controller = CrazyController(env)
    rewards = []

    print(f"--- Starting simulation loop (500 steps) ---")
    for i in range(500):
        
        terminated = truncated = False
        current_month = info['month']
        action = controller.act(observation)


        # you can check opengym and sinergym to understand this in more depth
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        #   --- BELOW GIVEN PRINT STATEMENTS can be used to check observation, logging, and reward data at each timestep ---
        print(f"Step: {i+1}/500, Reward: {reward:.4f}")
        #print(info) #uncomment this to see helpful logging metadata
        #print("OBSERVATION (the numbers):", observation) #uncomment this to see the observation values at each timestep

        if terminated or truncated:
            print(f"--- Episode finished at step {i+1}, resetting... ---")
            observation, info = env.reset()

    print("\n--- EPISODE FINISHED ---")
    print(f"Episode Mean reward: {np.mean(rewards):.4f}")
    print(f"Episode Cumulative reward: {sum(rewards):.2f}")
    print("--------------------------\n")

    env.close()
    print("--- Controller Script Finished ---")