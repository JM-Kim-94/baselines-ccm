import numpy as np
import random

# from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from gym.envs.mujoco import AntEnv as AntEnv

from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-dir-2개')
class AntDir2Env(AntEnv):
    def __init__(self, task={}, num_train_tasks=2, eval_tasks_list=[], randomize_tasks=True, env_type='train', **kwargs):
        self.env_type = env_type
        self._task = task
        self.n_train_tasks = num_train_tasks
        self.eval_tasks_list = eval_tasks_list

        self.tasks = self.sample_tasks()
        print("self.tasks", self.tasks)
        self._goal = self.tasks[0]['goal']

        super(AntDir2Env, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))
        direct = (np.cos(self._goal), np.sin(self._goal))
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))

        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self):
        if self.env_type == 'train':
            goal_dirs = [0.0,  0.5 * np.pi]

        elif self.env_type == 'test':
            goal_dirs = [1 * np.pi / 4,   3 * np.pi / 4,    7 * np.pi / 4]   # indi, extra, extra

        elif self.env_type == 'tsne':
            goal_dirs = [0.0 * np.pi / 4,
                         1.0 * np.pi / 4,
                         2.0 * np.pi / 4,
                         3.0 * np.pi / 4,
                         7.0 * np.pi / 4]  # 5개

        else:
            goal_dirs = None

        tasks = [{'goal': goal_dir} for goal_dir in goal_dirs]
        return tasks


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])


    def get_all_task_idx(self):
        return range(len(self.tasks))


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()


