import numpy as np
from rand_param_envs.walker2d_mass_inter import Walker2DMassInterEnv

from . import register_env


@register_env('walker-mass-inter')
class WalkerMassInterEnv(Walker2DMassInterEnv):
    def __init__(self, num_train_tasks, env_type, eval_tasks_list):
        super(WalkerMassInterEnv, self).__init__()
        self.tasks, self.tasks_value = self.sample_tasks(num_train_tasks, env_type, eval_tasks_list)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
