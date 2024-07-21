import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.core.latent_vectors import LatentVectors, EnvType
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu

from MulticoreTSNE import MulticoreTSNE as TSNE
# pip install git+https://github.com/jorvis/Multicore-TSNE

from datetime import datetime

import os
import matplotlib.pyplot as plt

import wandb
class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env_name,

            env,
            env_eval,
            env_tsne,
            agent1,
            agent2,
            train_tasks,
            eval_tasks,
            tsne_tasks,
            variant,

            use_information_bottleneck=True,
            log_dir="",
            num_tsne_evals=30,
            tsne_plot_freq=20,
            exp_name='',
            encoder_tau=0.005,
            alpha=1,
            kl_lambda=1.0,
            ce_coeff=1.0,
            latent_dim=7,
            meta_batch=64,
            num_iterations=100,
            num_pretrain_steps_per_itr=80000,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=100000,
            replay_buffer_size_exp=80000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            meta_episode_len=2,  # 10,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.use_information_bottleneck = use_information_bottleneck
        self.log_dir = log_dir
        self.num_tsne_evals = num_tsne_evals
        self.tsne_plot_freq = tsne_plot_freq
        self.tsne_tasks = tsne_tasks
        self.env_tsne = env_tsne
        self.exp_name = exp_name
        self.encoder_tau = encoder_tau
        self.alpha = alpha
        self.kl_lambda = kl_lambda
        self.ce_coeff = ce_coeff
        self.variant = variant
        self.env_name = env_name
        self.env = env
        self.env_eval = env_eval
        self.agent = agent1
        self.exploration_agent = agent2  # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.latent_dim = latent_dim
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_pretrain_steps_per_itr = num_pretrain_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.meta_episode_len = meta_episode_len
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.replay_buffer_size_exp = replay_buffer_size_exp
        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent1,
            explore_policy=agent2,
            max_path_length=self.max_path_length,
        )
        self.sampler_eval = InPlacePathSampler(
            env=env_eval,
            policy=agent1,
            explore_policy=agent2,
            max_path_length=self.max_path_length,
        )




        ###############################
        self.sampler_tsne = InPlacePathSampler(
            env=env_tsne,
            policy=agent1,
            explore_policy=agent2,
            max_path_length=self.max_path_length,
        )
        ###############################

        self.tsne_log_dir = f"logs/{self.env_name}/{datetime.now()}/tsne"
        os.makedirs(self.tsne_log_dir)

        kl = str(self.kl_lambda) if self.use_information_bottleneck else "X"

        wandb.login(key="7316f79887c82500a01a529518f2af73d5520255")
        wandb.init(
            # set the wandb project where this run will be logged
            entity="mlic_academic",
            project='김정모_metaRL_baselines',
            group=self.env_name,  # 'ccm',  # "pearl-antgoal",#self.env_name,
            name='ccm-' + self.env_name \
                 + '_exp' + self.exp_name \
                 + '_ce' + str(self.ce_coeff) \
                 + '_kl' + kl \
                 + '_rs' + str(self.reward_scale) \
                 + '_alpha' + str(self.alpha) \
                 + '_enctau' + str(self.encoder_tau)
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size * 2,
                env,
                self.train_tasks,
                latent_dim=self.latent_dim,
            )
        self.replay_buffer_exp = MultiTaskReplayBuffer(
                self.replay_buffer_size_exp,
                env,
                self.train_tasks,
                latent_dim=self.latent_dim,
        )
        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
                latent_dim=self.latent_dim,
        )


        self._n_env_steps_total = 0
        self._n_pretrain_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''

        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ > 0:
                explore = True
            else:
                explore = False
            if it_ == 0:

                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.replay_buffer_exp.task_buffers[idx].clear()
                    self.collect_data(self.num_initial_steps, 1, np.inf, explore=False, add_to_exp_buffer=False,
                                      add_to_buffer=True)

            if it_ == 1 or it_ == 31:
                
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.replay_buffer_exp.task_buffers[idx].clear()
                    self.collect_data(self.num_initial_steps, 1, np.inf, explore=True, add_to_exp_buffer=True, add_to_buffer=False)
            # Sample data from train tasks.
            if it_ == 0:
                print('pretraining...')
                for pretrain_step in range(self.num_pretrain_steps_per_itr):
                    pre_indices = np.random.choice(self.train_tasks, self.meta_batch)
                    self.pretrain(pre_indices)
                    self._n_pretrain_steps_total += 1
                    if pretrain_step % 10000 == 0:
                        print("pretrain step {} / {}".format(pretrain_step, self.num_pretrain_steps_per_itr))
                print('done for pretraining')

            print("collect data for {} tasks".format(self.num_tasks_sample))
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                print("idx", idx)
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, explore=False, add_to_exp_buffer=False, add_to_buffer=True)
                    self.collect_data(self.num_steps_prior, 1, np.inf, explore=explore, add_to_exp_buffer=explore, add_to_buffer=False)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, explore=False, add_to_exp_buffer=False, add_to_buffer=True)
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train, explore=explore, add_to_exp_buffer=explore, add_to_buffer=False)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False, explore=explore, add_to_exp_buffer=explore, add_to_buffer=False)
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False, explore=False, add_to_exp_buffer=False, add_to_buffer=True)
            # Sample train tasks and compute gradient updates on parameters.
            print("done for collect data")

            # losses = [[] for _ in range(4)]
            loss_dict = {"cross_entropy_loss": [],
                         "qf_loss": [],
                         "vf_loss": [],
                         "policy_loss": [],
                         "log_pi": []}
            exp_loss_dict = {"exp_qf_loss": [],
                             "exp_vf_loss": [],
                             "exp_policy_loss": [],
                             "exp_log_pi": []}

            print("main train start")
            for train_step in range(self.num_train_steps_per_itr):

                indices = np.random.choice(self.train_tasks, self.meta_batch)
                if it_ > 0:
                    loss, exp_loss = self._do_training(indices, exp=True)
                else:
                    loss, exp_loss = self._do_training(indices)

                for key in loss.keys():
                    loss_dict[key].append(loss[key])
                if exp_loss is not None:
                    for key in exp_loss.keys():
                        exp_loss_dict[key].append(exp_loss[key])
                else:
                    for key in exp_loss_dict.keys():
                        exp_loss_dict[key].append(0.0)

                self._n_train_steps_total += 1
            print("main train end")

            for key in loss_dict.keys():
                loss_dict[key] = np.mean(loss_dict[key])
            for key in exp_loss_dict.keys():
                exp_loss_dict[key] = np.mean(exp_loss_dict[key])

            gt.stamp('train')

            self.training_mode(False)

            # eval
            print("start evaluation")
            self._try_to_eval(it_, loss_dict, exp_loss_dict)
            gt.stamp('eval')
            print("end evaluation")

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True, explore=False, print_success=False, add_to_exp_buffer=True, add_to_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        mb_size = self.embedding_mini_batch_size
        self.agent.clear_z()
        indices = np.random.choice(self.train_tasks, self.meta_batch - 1)

        indices = np.insert(indices, 0, self.task_idx)
        num_transitions = 0
        while num_transitions < num_samples:
            if explore == True:
                batch = self.sample_data(indices, encoder=True)

                mini_batch = [x[:, : mb_size, :] for x in batch]
                # print('batch:', mini_batch)
                obs_enc, act_enc, rewards_enc, nobs_enc, _, rewards_exp_enc, z_previous_enc = mini_batch
                context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
                # print('context:', context)
                z_keys = self.agent.encode(context, ema=True)
                if update_posterior_rate == np.inf:
                    self.agent.clear_z()
            else:
                z_keys=None
            paths, n_samples, info = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                                max_trajs=1,
                                                                accum_context=False,
                                                                resample=resample_z_rate,
                                                                 explore=explore,
                                                                 context=z_keys)
            num_transitions += n_samples
            if add_to_buffer==True:
                self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_exp_buffer==True:
                self.replay_buffer_exp.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.prepare_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch, loss_dict, exp_loss_dict):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch, loss_dict, exp_loss_dict)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run, wideeval=False, explore=False, context=None):
        self.task_idx = idx
        if wideeval==False:
            self.env.reset_task(idx)
        else:
            self.env_eval.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            if explore==True:
                context = self.prepare_context(self.task_idx)

                z_keys=self.agent.encode(context, ema=True)
                self.agent.clear_z()
            else:
                z_keys=None
            if wideeval==False:
                path, num, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1, accum_context=True, explore=explore, context=z_keys)
            else:
                path, num, info = self.sampler_eval.obtain_samples(deterministic=self.eval_deterministic,
                                                              max_samples=self.num_steps_per_eval - num_transitions,
                                                              max_trajs=1, accum_context=True, explore=explore, context=z_keys)
            num_trajs += 1
            paths += path
            num_transitions += num
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def collect_paths_exp(self, idx, epoch, run, wideeval=False, explore=False, context=None, return_z=False, tsne=False):
        # print("self.meta_episode_len", self.meta_episode_len)
        self.task_idx = idx
        if wideeval==False:
            self.env.reset_task(idx)
        else:
            if tsne:
                self.env_tsne.reset_task(idx)
            else:
                self.env_eval.reset_task(idx)

        indices = np.random.choice(self.train_tasks, self.meta_batch)
        self.agent.clear_z(num_tasks=len(indices))  # --> z~Normal or 1 //  ctxt=None
        paths = []
        num_transitions = 0
        num_trajs = 0
        #context = self.prepare_context(self.task_idx)

        z_keys = self.agent.z  # z_keys~Normal or 1
        self.agent.clear_z()
        if wideeval == False:
            path, num, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                          max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys)
        else:
            if tsne:
                path, num, info = self.sampler_tsne.obtain_samples(deterministic=self.eval_deterministic,
                                                                   max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys)
            else:
                path, num, info = self.sampler_eval.obtain_samples(deterministic=self.eval_deterministic,
                                                               max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys)
        # """accum_context=True 이게 있어야 되는거 아님?"""
        # if wideeval == False:
        #     path, num, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
        #                                                   max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys, accum_context=True)
        # else:
        #     if tsne:
        #         path, num, info = self.sampler_tsne.obtain_samples(deterministic=self.eval_deterministic,
        #                                                            max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys, accum_context=True)
        #     else:
        #         path, num, info = self.sampler_eval.obtain_samples(deterministic=self.eval_deterministic,
        #                                                            max_trajs=self.meta_episode_len, explore=True, infer=False, context=z_keys, accum_context=True)

        num_trajs += self.meta_episode_len
        paths += path
        num_transitions += num
        self.agent.infer_posterior(self.agent.context)
        task_z = self.agent.z  # for TSNE

        print("self.num_steps_per_eval", self.num_steps_per_eval)
        print("num_transitions", num_transitions)

        if wideeval == False:
            path, num, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                          max_samples=self.num_steps_per_eval - num_transitions, infer=True, accum_context=True)
            # task_z = None
        else:
            if tsne:
                path, num, info = self.sampler_tsne.obtain_samples(deterministic=self.eval_deterministic,
                                                                           max_samples=self.num_steps_per_eval - num_transitions,
                                                                           infer=True, accum_context=True,
                                                                           return_z=False)
            else:
                path, num, info = self.sampler_eval.obtain_samples(deterministic=self.eval_deterministic,
                                                                           max_samples=self.num_steps_per_eval - num_transitions,
                                                                           infer=True, accum_context=True,
                                                                           return_z=False)
        # print("task_z", task_z.shape)
        num_transitions += num
        num_trajs += 1
        paths += path
        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards


        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        if return_z:
            return paths, task_z
        else:
            return paths


    def _do_eval(self, indices, epoch, wideeval=False, tsne=False):
        final_returns = []
        online_returns = []
        final_returns_last = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths_exp(idx, epoch, r, wideeval=wideeval, tsne=tsne)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])

                print("paths", len(paths))

            final_returns_last.append(np.mean([a[-1] for a in all_rets]))
            final_returns.append(np.mean([np.mean(a) for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns, final_returns_last

    def get_l2_distance_matrix(self, z_lst):
        length = len(z_lst)
        m = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                diff = z_lst[i] - z_lst[j]
                euclidian = np.linalg.norm(diff, ord=2)
                m[i, j] = euclidian
        return m

    def _do_tsne_eval_add_inter_plot(self, indices, epoch):

        plt.figure(figsize=(12, 5))

        z_total_list = [[] for _ in range(len(indices))]  # z tsne에 필요
        z_list = []  # 디스턴스 매트릭스 계산시 필요
        for label, task_idx in enumerate(indices):

            one_task_z_ = []
            for run in range(self.num_tsne_evals):  # 30번
                paths, z = self.collect_paths_exp(task_idx, epoch, run, wideeval=True, return_z=True)
                z = z.view(-1, ).detach().cpu().numpy()  # z (10,)

                z_total_list[label].append(z)
                one_task_z_.append(z)  # 디스턴스 매트릭스 계산시
            z_list.append(sum(one_task_z_) / len(one_task_z_))  # 디스턴스 매트릭스 계산시 필요

        distance_matrix = self.get_l2_distance_matrix(z_list)

        z_total_list_flatten = []
        for i in range(len(z_total_list)):  # 17
            for j in range(len(z_total_list[i])):
                z_total_list_flatten.append(z_total_list[i][j])

        indices_list = [range(i * self.num_tsne_evals, (i + 1) * self.num_tsne_evals) for i in range(len(z_total_list))]

        tsne_model = TSNE(n_components=2, random_state=0, perplexity=50, n_jobs=4)
        result = tsne_model.fit_transform(np.array(z_total_list_flatten))

        if self.env_name == "cheetah-vel-inter":
            tsne_tasks_lst = ["0.1(c0)", "0.25(c1)", "0.75(c2)", "1.25(c3)",
                              "1.75(c4)", "2.25(c5)", "2.75(c6)", "3.1(c7)", "3.25(c8)"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$"]

        elif self.env_name == "ant-goal-inter":
            tsne_tasks_lst = ["[0.5,  0]c0", "[0, 0.5 ]c1", "[-0.5,  0]c2", "[0, -0.5 ]c3",
                              "[1.75, 0]c4", "[0, 1.75]c5", "[-1.75, 0]c6", "[0, -1.75]c7",
                              "[2.75, 0]c8", "[0, 2.75]c9", "[-2.75, 0]c10", '[0, -2.75]c11']
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$", "$c_{9}$",
                           "$c_{10}$", "$c_{11}$"]

        elif self.env_name == "ant-dir-4개":
            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3",
                              "4/4pi c4", "5/4pi c5", "6/4pi c6", "7/4pi c7"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$"]

        elif self.env_name == "ant-dir-2개":
            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3", "7/4pi c4"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$"]

        elif "mass" in self.env_name or "params" in self.env_name:
            tsne_tasks_lst = ["c" + str(i) for i in range(len(self.tsne_tasks))]
            marker_list = ["$c_{" + str(i) + "}$" for i in range(len(self.tsne_tasks))]

        else:
            tsne_tasks_lst, marker_list = None, None

        plt.subplot(121)
        for i in range(len(z_total_list)):
            plt.scatter(result[:, 0][indices_list[i]],
                        result[:, 1][indices_list[i]],
                        s=100, marker=marker_list[i],  # c=colors[i], # 's',
                        # alpha=0.02, edgecolor='k',
                        label=str(tsne_tasks_lst[i]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Task latent variable(c) T-SNE plot")

        plt.subplot(122)
        for i in range(len(distance_matrix)):
            text = 'c_' + str(i)
            plt.text(i, -0.7, text, rotation=90)
            plt.text(-1.5, i, text)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.imshow(distance_matrix)
        plt.colorbar()

        tsne_save_path = os.path.join(self.log_dir, "-tSNE_" + str(epoch) + '.png')
        plt.savefig(tsne_save_path, bbox_inches='tight')

        wandb.log({
            "eval_tsne/tsne_ep" + str(epoch): [wandb.Image(tsne_save_path)]
        })


    def _do_tsne_plot(self, epoch):
        trials = 30
        latent_samples = []
        indices_list = [[] for _ in self.tsne_tasks]
        i = 0

        for trial in range(trials):
            for task in self.tsne_tasks:
                _, latent_sample = self.collect_paths_exp(task, epoch, trial, wideeval=True, return_z=True, tsne=True)
                latent_samples.append(latent_sample.squeeze().numpy(force=True))
                indices_list[task].append(i)
                i += 1

        latent_vectors = LatentVectors(np.asarray(latent_samples), indices_list, epoch, EnvType.parse(self.env_name))
        latent_vectors.save(self.tsne_log_dir)
        latent_vectors.save_plot(self.tsne_log_dir, wandb)



    def evaluate(self, epoch, loss_dict, exp_loss_dict):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        # if self.dump_eval_paths:
        #     # 100 arbitrarily chosen for visualizations of point_robot trajectories
        #     # just want stochasticity of z, not the policy
        #     self.agent.clear_z()
        #     prior_paths, _, info1 = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
        #                                                 accum_context=False,
        #                                                 resample=1)
        #     logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        print("start train task eval on ", indices)
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        train_suc = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            a_s = 0
            for _ in range(self.num_steps_per_eval // self.max_path_length):  # 600//200 3
                context = self.prepare_context(idx)  # ~ enc_buffer
                self.agent.infer_posterior(context)
                p, _, info = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length,
                                                        accum_context=False,
                                                        max_trajs=1,
                                                        resample=np.inf)
                a_s += info['n_success_num']
                paths += p
            a_s = a_s / (self.num_steps_per_eval // self.max_path_length)
            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards
            #print(paths)

            train_returns.append(eval_util.get_average_returns(paths))
            train_suc.append(a_s)
            #print(train_returns)
        train_returns = np.mean(train_returns)
        print("train_returns", train_returns)
        train_suc = np.mean(train_suc)
        #print(train_returns)

        print("start train task evaluation")
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns, train_final_returns_last = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        print("start test task evaluation")
        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns, test_final_returns_last = self._do_eval(self.eval_tasks, epoch, wideeval=True)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)



        """TSNE"""
        if epoch % self.tsne_plot_freq == 0:
            # print("my tsne plot")
            # self._do_tsne_eval_add_inter_plot(self.tsne_tasks, epoch)
            # # if epoch % self.tsne_plot_freq == 0:
            self.traintask_tsne(self.eval_tasks, epoch)

            print("wide tsne plot")
            self._do_tsne_plot(epoch)

            print("tsne plot end")





        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        #if hasattr(self.env, "log_diagnostics"):
        #    self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_return_last = np.mean(train_final_returns_last)
        avg_test_return_last = np.mean(test_final_returns_last)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        self.eval_statistics['AverageReturn_all_train_tasks_last'] = avg_train_return_last
        self.eval_statistics['AverageReturn_all_test_tasks_last'] = avg_test_return_last
        self.eval_statistics['Averagesuc_rate'] = train_suc
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        wandb_log_dict = {
            "Eval/train_avg_return": avg_train_return,  # avg_train_return_last,
            "Eval/test_avg_return": avg_test_return,  # avg_test_return_last,
            # "Eval/train_avg_return": avg_train_return,
            # "Eval/test_avg_return": avg_test_return,
            "AverageTrainReturn_all_train_tasks": train_returns,
            "AverageReturn_all_train_tasks": avg_train_return,
            "AverageReturn_all_test_tasks": avg_test_return,
            "AverageReturn_all_train_tasks_last": avg_train_return_last,
            "AverageReturn_all_test_tasks_last": avg_test_return_last,
        }
        wandb_log_dict.update(loss_dict)
        wandb_log_dict.update(exp_loss_dict)
        env_step = self._n_env_steps_total
        wandb.log(wandb_log_dict, step=env_step)


        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    def traintask_tsne(self, indices, epoch):
        # indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        z_list = []
        for idx in indices:
            for r in range(30):  # 600//200 3
                # context = self.prepare_context(idx)  # ~ enc_buffer
                # self.agent.infer_posterior(context)
                # z_list.append(self.agent.z)

                paths, z = self.collect_paths_exp(idx, epoch, r, wideeval=True, return_z=True)
                z = z.view(-1, ).detach().cpu().numpy()  # z (10,)
                z_list.append(z)


        plt.figure(figsize=(12, 5))

        z_total_list = [[] for _ in range(len(indices))]  # z tsne에 필요
        z_list = []  # 디스턴스 매트릭스 계산시 필요
        for label, task_idx in enumerate(indices):

            one_task_z_ = []
            for run in range(self.num_tsne_evals):  # 30번
                context = self.prepare_context(task_idx)  # ~ enc_buffer
                self.agent.infer_posterior(context)
                z = self.agent.z
                z = z.view(-1, ).detach().cpu().numpy()  # z (10,)

                z_total_list[label].append(z)
                one_task_z_.append(z)  # 디스턴스 매트릭스 계산시
            z_list.append(sum(one_task_z_) / len(one_task_z_))  # 디스턴스 매트릭스 계산시 필요

        distance_matrix = self.get_l2_distance_matrix(z_list)

        z_total_list_flatten = []
        for i in range(len(z_total_list)):  # 17
            for j in range(len(z_total_list[i])):
                z_total_list_flatten.append(z_total_list[i][j])

        indices_list = [range(i * self.num_tsne_evals, (i + 1) * self.num_tsne_evals) for i in range(len(z_total_list))]

        tsne_model = TSNE(n_components=2, random_state=0, perplexity=50, n_jobs=4)
        result = tsne_model.fit_transform(np.array(z_total_list_flatten))

        if self.env_name == "cheetah-vel-inter":
            tsne_tasks_lst = ["0.1(c0)", "0.25(c1)", "0.75(c2)", "1.25(c3)",
                              "1.75(c4)", "2.25(c5)", "2.75(c6)", "3.1(c7)", "3.25(c8)"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$", "$c_8$"]

        elif self.env_name == "ant-goal-inter":
            # [[1.75, 0], [0, 1.75], [-1.75, 0], [0, -1.75]]
            tsne_tasks_lst = ["[1.75, 0]c0", "[0, 1.75]c1", "[-1.75, 0]c2", "[0, -1.75]c3"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$"]

        elif self.env_name == "ant-dir-4개":
            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3",
                              "4/4pi c4", "5/4pi c5", "6/4pi c6", "7/4pi c7"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$", "$c_5$", "$c_6$", "$c_7$"]

        elif self.env_name == "ant-dir-2개":
            tsne_tasks_lst = ["0pi c0", "1/4pi c1", "2/4pic2", "3/4pi c3", "7/4pi c4"]
            marker_list = ["$c_0$", "$c_1$", "$c_2$", "$c_3$", "$c_4$"]

        elif "mass" in self.env_name or "params" in self.env_name:
            tsne_tasks_lst = ["c" + str(i) for i in range(len(self.tsne_tasks))]
            marker_list = ["$c_{" + str(i) + "}$" for i in range(len(self.tsne_tasks))]

        else:
            tsne_tasks_lst, marker_list = None, None

        plt.subplot(121)
        for i in range(len(z_total_list)):
            plt.scatter(result[:, 0][indices_list[i]],
                        result[:, 1][indices_list[i]],
                        s=100, marker=marker_list[i],  # c=colors[i], # 's',
                        # alpha=0.02, edgecolor='k',
                        label=str(tsne_tasks_lst[i]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Task latent variable(c) T-SNE plot")

        plt.subplot(122)
        for i in range(len(distance_matrix)):
            text = 'c_' + str(i)
            plt.text(i, -0.7, text, rotation=90)
            plt.text(-1.5, i, text)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.imshow(distance_matrix)
        plt.colorbar()

        tsne_save_path = os.path.join(self.log_dir, "-tSNE_" + str(epoch) + '.png')
        # plt.savefig(tsne_save_path, bbox_inches='tight')
        plt.savefig(tsne_save_path)

        wandb.log({
            "eval_tsne/tsne_ep" + str(epoch): [wandb.Image(tsne_save_path)]
        })





    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

