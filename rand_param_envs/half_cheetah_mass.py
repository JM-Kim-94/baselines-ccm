



import numpy as np
from rand_param_envs.base2 import RandomEnv
from rand_param_envs.gym import utils

# from . import register_env

# @register_env('cheetah-mass')
class HalfCheetahMassEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'half_cheetah.xml')
        utils.EzPickle.__init__(self)
    
    def _step(self, action):
        xposbefore = self.model.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0]

        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False

        return ob, reward, done, {}

    
    def _get_obs(self):
        return np.concatenate(
            [
                self.model.data.qpos.flat[1:],
                self.model.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5
