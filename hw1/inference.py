import os
import gym
import torch
import pickle 
import numpy as np

from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import cs285.infrastructure.utils as utils

TIMESTEP = 1000
EVAL_BATCH_SIZE = 5000
EP_LEN = 1000
ENV_NAME = "Humanoid-v2"
POLICY_FN = "policy_itr_0.pt"
RENDER_MODE = None # "human", "rgb"
SEED = 0
agent_path = "data/q1_bc_human_iter_2500_Humanoid-v2_01-01-2023_16-18-46"
expert_path = "cs285/policies/experts/Humanoid.pkl"


env = gym.make(ENV_NAME, render_mode=RENDER_MODE)

expert = LoadedGaussianPolicy(expert_path)
agent_param = pickle.load(open(os.path.join(agent_path, "param.pkl"), "rb"))
agent = MLPPolicySL(
    env.action_space.shape[0], 
    env.observation_space.shape[0], 
    n_layers=agent_param["n_layers"],
    size=agent_param["size"]
)
agent.load_state_dict(torch.load(os.path.join(agent_path, POLICY_FN)))

eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
    env, agent, EVAL_BATCH_SIZE, EP_LEN, seed=SEED
)

expert_paths, expert_envsteps_this_batch = utils.sample_trajectories(
    env, expert, EVAL_BATCH_SIZE, EP_LEN, seed=SEED
)

eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
expert_returns = [expert_path["reward"].sum() for expert_path in expert_paths]

print("Imitated Learning Model:")
print("Mean of Return:", np.mean(eval_returns))
print("Std of Return:", np.std(eval_returns))

print("Expert Model:")
print("Mean of Return:", np.mean(expert_returns))
print("Std of Return:", np.std(expert_returns))
print("30% performance:", np.mean(expert_returns)*0.3)

# agent.eval()
# ob = env.reset()[0]
# for step in range(TIMESTEP):
#     action = agent.get_action
#     ob, rew, done, _, _, = env.step(action)
#     env.render()
#     if done:
#         break
