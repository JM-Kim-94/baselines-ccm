import numpy as np
import random

# from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from gym.envs.mujoco import AntEnv as AntEnv

from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal-inter')
class AntGoalInterEnv(AntEnv):
    def __init__(self, task={}, num_train_tasks=2, eval_tasks_list=[], randomize_tasks=True, env_type='train', **kwargs):
        self.env_type = env_type
        self._task = task
        self.n_train_tasks = num_train_tasks
        self.eval_tasks_list = eval_tasks_list

        self.tasks = self.sample_tasks()
        self._goal = self.tasks[0]['goal']

        super(AntGoalInterEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            done_g=xposafter[:2],
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self):
        # a = np.random.random(num_tasks) * 2 * np.pi
        # r = 3 * np.random.random(num_tasks) ** 0.5
        # goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        # tasks = [{'goal': goal} for goal in goals]
        # return tasks
        if self.env_type == 'train':
            goals = []
            for i in range(self.n_train_tasks):
                prob = random.random()
                if prob < 4.0 / 15.0:
                    r = random.random() ** 0.5  # [0, 1]
                else:
                    r = (random.random() * 2.75 + 6.25) ** 0.5
                theta = random.random() * 2 * np.pi  # [0.0, 2pi]
                goals.append([r * np.cos(theta), r * np.sin(theta)])

        elif self.env_type == 'test':

            if len(self.eval_tasks_list) > 0:
                goals = self.eval_tasks_list
            else:
                theta_list = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * np.pi / 4
                train_r_list = np.array([0.5, 1.0, 2.5, 3.0])
                test_r_list = np.array([1.5, 2.0])
                train_tsne_tasks_list, test_tsne_tasks_list = [], []

                for r in train_r_list:
                    for theta in theta_list:
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        train_tsne_tasks_list.append([x, y])

                for r in test_r_list:
                    for theta in theta_list:
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        test_tsne_tasks_list.append([x, y])

                goals = train_tsne_tasks_list + test_tsne_tasks_list

        else:
            goals = None

        tasks = [{'goal': goal} for goal in goals]
        return tasks


    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat,
    #         self.sim.data.qvel.flat,
    #         np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    #     ])
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


    def get_all_task_idx(self):
        return range(len(self.tasks))


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()


