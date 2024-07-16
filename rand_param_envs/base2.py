
from rand_param_envs.gym.core import Env
from rand_param_envs.gym.envs.mujoco import MujocoEnv
import numpy as np
import random


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks, env_type, eval_list):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass


class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, rand_params=RAND_PARAMS, **kwargs):
        # print("log_scale_limit", log_scale_limit)  # log_scale_limit 3.0
        # print("file_name", file_name)  # file_name walker2d.xml
        # print("*args", *args)  # *args 5
        # print("rand_params", rand_params)  # rand_params ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
        # print("**kwargs", **kwargs)  # **kwargs
        print("file_name in base2.py", file_name)

        MujocoEnv.__init__(self, file_name, 4)
        # assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
        #     "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit
        self.rand_params = rand_params
        self.save_parameters()

    def get_one_rand_params(self, eval_mode='train', value=0):
        new_params = {}

        mass_size_ = np.prod(self.model.body_mass.shape)
        if eval_mode == "train":
            prob = random.random()
            if prob >= 0.5:
                body_mass_multiplyers_ = random.uniform(0, 0.5)
            else:
                body_mass_multiplyers_ = random.uniform(3.0, 3.5)  # 3.0 - 0.5 = 2.5

        elif eval_mode == "eval":
            body_mass_multiplyers_ = value

        else:
            body_mass_multiplyers_ = None

        print("body_mass_multiplyers_ in base2", body_mass_multiplyers_)

        body_mass_multiplyers = np.array([body_mass_multiplyers_ for _ in range(mass_size_)])
        body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
        body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
        new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

        return new_params, body_mass_multiplyers_

    # def sample_tasks(self, num_train_tasks, env_type, eval_tasks_list):
    #
    #     train_tasks, eval_tasks = [], []
    #     train_tasks_value, eval_tasks_value = [], []
    #
    #     # get_one_rand_params(self, task='inter', eval_mode='train', value=0):
    #     """train_task"""  # [0,0.5] + [3,3.5]
    #     for _ in range(num_train_tasks):  #
    #         new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='train')
    #         train_tasks.append(new_params)
    #         train_tasks_value.append(body_mass_multiplyers_)
    #
    #     """eval_task"""  #
    #     for v in eval_tasks_list:
    #         new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='eval', value=v)
    #         eval_tasks.append(new_params)
    #         eval_tasks_value.append(body_mass_multiplyers_)
    #
    #     """total tasks list"""
    #     param_sets = train_tasks + eval_tasks
    #     param_sets_value_list = train_tasks_value + eval_tasks_value
    #     return param_sets, param_sets_value_list

    def sample_tasks(self, num_train_tasks, env_type, eval_tasks_list):

        tasks, tasks_value = [], []

        if env_type == "train":  # [0,0.5] + [3,3.5]
            for _ in range(num_train_tasks):  #
                new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='train')
                # print("body_mass_multiplyers_ in sample_tasks envtype=train : ", body_mass_multiplyers_)
                tasks.append(new_params)
                tasks_value.append(body_mass_multiplyers_)

        elif env_type == "test" and len(eval_tasks_list) > 0:
            for v in eval_tasks_list:
                new_params, body_mass_multiplyers_ = self.get_one_rand_params(eval_mode='eval', value=v)
                # print("body_mass_multiplyers_ in sample_tasks envtype=test : ", body_mass_multiplyers_)
                tasks.append(new_params)
                tasks_value.append(body_mass_multiplyers_)

        return tasks, tasks_value

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        self.init_params['body_mass'] = self.model.body_mass

        self.cur_params = self.init_params