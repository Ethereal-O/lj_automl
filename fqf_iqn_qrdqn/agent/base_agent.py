from abc import ABC, abstractmethod
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fqf_iqn_qrdqn.memory import LazyMultiStepMemory, \
    LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import RunningMeanStats, LinearAnneaer


class BaseAgent(ABC):

    def __init__(self, env, valid_calculator, test_calculator, log_dir, num_steps=5*(10**7),
                 batch_size=32, memory_size=10**6, gamma=0.99, multi_step=1,
                 update_interval=4, target_update_interval=10000,
                 start_steps=50000, epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=5.0, cuda=True, seed=0,
                 num_parallel_envs=1):  # 新增：并行环境数量

        # 暂存待补全reward的transitions
        self.pending_transitions = []  # 格式: dict with state, action, next_state, done, reward_type, episode_ref

        self.env = env
        self.num_parallel_envs = num_parallel_envs
        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator

        torch.manual_seed(seed)
        np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.online_net = None
        self.target_net = None
        self.online_mean_net = None
        self.target_mean_net = None

        # Replay memory which is memory-efficient to store stacked frames.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step, beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.best_test_score = -np.inf
        self.num_actions = self.env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.double_q_learning = double_q_learning
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net
        self.use_per = use_per

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.epsilon_train = LinearAnneaer(1.0, epsilon_train, epsilon_decay_steps)
        self.epsilon_eval = epsilon_eval
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps\
            and len(self.memory) >= self.batch_size

    def is_random(self, eval=False):
        # Use e-greedy for evaluation.
        if self.steps < self.start_steps:
            return True
        if eval:
            return np.random.rand() < self.epsilon_eval
        if self.noisy_net:
            return False

        # 语法学习阶段增强随机性，鼓励探索多字段表达式
        import os
        is_syntax_learning = os.environ.get('ALPHAQCM_SYNTAX_LEARNING', '').lower() == 'true'
        if is_syntax_learning:
            # 语法学习阶段：大幅提高随机行动概率至50%
            return np.random.rand() < 0.5  # 固定50%随机性
        else:
            # IC学习阶段：正常epsilon
            return np.random.rand() < self.epsilon_train.get()

    def update_target(self):
        self.target_net.load_state_dict(
            self.online_net.state_dict())

    def explore(self):
        # Act with randomness.
        allowed_action = self.env.action_masks()

        # 移除强制字段选择的限制，让RL自由探索

        # 特殊处理SEP动作：当SEP可用时，给它更高的选择概率
        if hasattr(self.env, 'sep_action') and self.env.sep_action is not None:
            sep_idx = self.env.sep_action
            if allowed_action[sep_idx]:
                # SEP可用时，有30%的概率直接选择SEP
                if np.random.rand() < 0.3:
                    return sep_idx

        # 正常随机选择（可能使用受限的动作空间）
        action = self.env.action_space.sample()
        while not allowed_action[action]:
            action = self.env.action_space.sample()
        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(state).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            q_values = self.online_net.calculate_q(states=state)
            forbid_action = torch.BoolTensor(
                ~self.env.action_masks()).to(self.device)
            q_values[:, forbid_action] = -1e6
            action = q_values.argmax().item()
        return action

    @abstractmethod
    def learn(self):
        """Learn from experience. Must be implemented by subclasses."""
        raise NotImplementedError("learn() method must be implemented by subclasses")

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(
            self.online_net.state_dict(),
            os.path.join(save_dir, 'online_net.pth'))
        torch.save(
            self.target_net.state_dict(),
            os.path.join(save_dir, 'target_net.pth'))
        if (self.online_mean_net is not None) & (self.target_mean_net is not None):
            torch.save(
                self.online_mean_net.state_dict(),
                os.path.join(save_dir, 'online_mean_net.pth'))
            torch.save(
                self.target_mean_net.state_dict(),
                os.path.join(save_dir, 'target_mean_net.pth'))
        

    def load_models(self, save_dir, require_mean = False):
        self.online_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'online_net.pth')))
        self.target_net.load_state_dict(torch.load(
            os.path.join(save_dir, 'target_net.pth')))
        
        if require_mean:
            self.online_net.load_state_dict(torch.load(
                os.path.join(save_dir, 'online_mean_net.pth')))
            self.target_net.load_state_dict(torch.load(
                os.path.join(save_dir, 'target_mean_net.pth')))

    def save_exprs(self, save_dir, valid_ic, test_ic, set_indice):
        state = self.env.pool.state
        n = len(state['exprs'])

        log_table = pd.DataFrame(
            columns=['exprs', 'ic', 'weight'], index=range(n + 1))

        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]

            log_table.loc[i, :] = (expr_str, ic_ret, weight)

        if set_indice == 'test':
            
            log_table.loc[n, :] = ('Ensemble', valid_ic, test_ic)
            log_table.to_csv(f'{save_dir}/test_best_table.csv')
        elif set_indice == 'valid':
            log_table.loc[n, :] = ('Ensemble', valid_ic, test_ic)
            log_table.to_csv(f'{save_dir}/valid_best_table.csv')

    # def save_agent(self, save_dir):
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     torch.save({'train_return': self.train_return, 'steps': self.steps,
    #                 'learning_steps': self.learning_steps, 'episodes': self.episodes,
    #                 'best_eval_score': self.best_eval_score, 'epsilon_train': self.epsilon_train,
    #                 'optim_online': self.optim_online.state_dict()},
    #                os.path.join(save_dir, 'agent.pkl'))

    # def load_agent(self, save_dir):
    #     checkpoint = torch.load(os.path.join(save_dir, 'agent.pkl'))
    #     self.train_return = checkpoint['train_return']
    #     self.steps = checkpoint['steps']
    #     self.learning_steps = checkpoint['learning_steps']
    #     self.episodes = checkpoint['episodes']
    #     self.best_eval_score = checkpoint['best_eval_score']
    #     self.epsilon_train = checkpoint['epsilon_train']
    #     self.optim_online.load_state_dict(checkpoint['optim_online'])

    def train_episode(self):
        self.online_net.train()
        self.target_net.train()

        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        try:
            state, info = self.env.reset()
        except Exception as e:
            print(f"❌ Failed to reset environment: {e}")
            import traceback
            traceback.print_exc()
            return  # Skip this episode

        while (not done) and episode_steps <= self.max_episode_steps:
            try:
                self.online_net.sample_noise()

                if self.is_random(eval=False):
                    action = self.explore()
                else:
                    action = self.exploit(state)

                next_state, reward, done, _, info = self.env.step(action)

                self.memory.append(state, action, reward, next_state, done)

                self.steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                self.train_step_interval()

            except Exception as e:
                print(f"❌ Error during training step {episode_steps}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with next episode...")
                episode_return -= 10.0
                break  # End this episode

        # Only print episode summary without step details

        # We log running mean of stats.
        self.train_return.append(episode_return)

        # We log evaluation results along with training steps.
        if self.episodes % self.log_interval == 0:
            try:
                self.writer.add_scalar(
                    'ic/train', self.env.env.pool.state['best_ic_ret'], self.steps)
            except Exception as e:
                print(f"Warning: Failed to log training stats: {e}")

    def train_step_interval(self):
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        # 处理所有已完成的异步reward计算
        self._process_completed_async_rewards()

        if self.is_update():
            self.learn()

    def _complete_pending_transitions(self, episode, final_reward, reward_type):
        """补全指定episode的pending transitions"""
        # 找到该episode的所有pending transitions
        transitions_to_complete = [
            t for t in self.pending_transitions
            if t['episode_ref'] is episode and t['reward_type'] == reward_type
        ]

        # 补全并保存到Replay Memory
        for transition in transitions_to_complete:
            self.memory.append(
                transition['state'],
                transition['action'],
                final_reward,  # 使用计算出的最终reward
                transition['next_state'],
                transition['done']
            )

            # 从pending列表中移除
            self.pending_transitions.remove(transition)

    def _process_completed_async_rewards(self):
        """处理所有已完成的异步reward计算"""
        # 这里可以添加额外的异步reward处理逻辑
        # 目前主要通过_complete_pending_transitions处理
        pass

    def evaluate(self):
        try:
            valid_ic = self.env.env.pool.test_ensemble(self.valid_calculator)
            test_ic = self.env.env.pool.test_ensemble(self.test_calculator)

            if valid_ic > self.best_eval_score:
                self.best_eval_score = valid_ic
                self.save_models(os.path.join(self.model_dir, 'best'))
                self.save_exprs(self.log_dir, valid_ic, test_ic, 'valid')

            if test_ic > self.best_test_score:
                self.best_test_score = test_ic
                self.save_exprs(self.log_dir, valid_ic, test_ic, 'test')

            # We log evaluation results along with training steps.
            self.writer.add_scalar('ic/valid', valid_ic, self.steps)
            self.writer.add_scalar('ic/test', test_ic, self.steps)

        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            # Continue training even if evaluation fails

    def __del__(self):
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except:
            pass  # Ignore errors during cleanup

        try:
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.close()
        except:
            pass  # Ignore errors during cleanup


class BatchAgent(BaseAgent):
    """支持多episode批处理的Agent"""

    def __init__(self, env_template, valid_calculator, test_calculator, log_dir,
                 num_parallel_envs=4, batch_final_threshold=None, **kwargs):
        # 创建多个环境的副本
        self.env_template = env_template
        self.envs = [self._create_env_copy() for _ in range(num_parallel_envs)]
        self.num_parallel_envs = num_parallel_envs

        # 状态2批量计算相关
        # 默认与并行环境数量一致，最小化延迟
        self.batch_final_threshold = batch_final_threshold or num_parallel_envs
        self.pending_final_expressions = []  # 等待批量计算的状态2表达式

        # 使用第一个环境作为代表进行初始化
        super().__init__(self.envs[0], valid_calculator, test_calculator, log_dir,
                        num_parallel_envs=num_parallel_envs, **kwargs)

    def _create_env_copy(self):
        """创建环境的独立副本"""
        # 这里需要实现环境的深拷贝或独立实例化
        # 目前简化处理，返回模板环境（在实际实现中需要更复杂的逻辑）
        return self.env_template

    def run(self):
        """批处理训练主循环"""
        while True:
            self.train_batch_episodes()
            if self.steps > self.num_steps:
                break

    def train_batch_episodes(self):
        """并行训练多个episodes"""
        # 初始化所有episodes
        episode_states = []
        for i, env in enumerate(self.envs):
            try:
                state, info = env.reset()
                episode_states.append({
                    'env_idx': i,
                    'env': env,
                    'state': state,
                    'done': False,
                    'return': 0.0,
                    'steps': 0,
                    'status': 'running'  # running, waiting_intermediate, waiting_final, terminated
                })
            except Exception as e:
                print(f"❌ Failed to reset env {i}: {e}")
                continue

        active_episodes = episode_states.copy()

        while active_episodes:
            # 阶段1: 并行执行所有活跃episodes
            completed_episodes = []
            waiting_intermediate = []
            waiting_final = []
            terminated_episodes = []

            for episode in active_episodes[:]:  # 复制列表以便修改
                if episode['status'] == 'running':
                    try:
                        # 选择动作
                        if self.is_random(eval=False):
                            action = self.explore_for_env(episode['env'])
                        else:
                            action = self.exploit_for_env(episode['env'], episode['state'])

                        # 执行动作
                        next_state, reward, done, truncated, info = episode['env'].step(action)

                        # 判断reward是否可以立即确定并保存transition
                        needs_async_reward = (done and not info.get('terminated_by_invalid')) or info.get('waiting_for_ic')

                        if needs_async_reward:
                            # 需要异步计算reward：暂存transition，等待补全
                            pending_transition = {
                                'episode_idx': episode['env_idx'],
                                'state': episode['state'],
                                'action': action,
                                'next_state': next_state,
                                'done': done,
                                'reward_type': 'waiting_final' if done else 'waiting_intermediate',
                                'episode_ref': episode  # 引用episode以便后续更新
                            }
                            self.pending_transitions.append(pending_transition)
                        else:
                            # reward可以立即确定：直接保存到Replay Memory
                            self.memory.append(episode['state'], action, reward, next_state, done)

                        # 更新状态
                        self.steps += 1
                        episode['steps'] += 1
                        episode['return'] += reward
                        episode['state'] = next_state

                        # 检查是否到达同步点
                        if done:
                            if info.get('terminated_by_invalid'):
                                # 状态1: 无效动作，无法挽回
                                episode['status'] = 'terminated'
                                terminated_episodes.append(episode)
                                completed_episodes.append(episode)
                            else:
                                # 状态2: 生成完成，到末尾表达式
                                episode['status'] = 'waiting_final'
                                waiting_final.append(episode)
                        elif info.get('waiting_for_ic'):
                            # 状态3: 生成到中间表达式
                            episode['status'] = 'waiting_intermediate'
                            waiting_intermediate.append(episode)

                        # 检查episode长度限制
                        if episode['steps'] >= self.max_episode_steps:
                            episode['status'] = 'terminated'
                            terminated_episodes.append(episode)
                            completed_episodes.append(episode)

                    except Exception as e:
                        print(f"❌ Error in episode {episode['env_idx']}: {e}")
                        episode['return'] -= 10.0
                        episode['status'] = 'terminated'
                        terminated_episodes.append(episode)
                        completed_episodes.append(episode)

            # 阶段2: 处理状态1（直接结束）和收集等待状态
            for ep in active_episodes[:]:  # 复制列表以便修改
                if ep['status'] == 'terminated':
                    terminated_episodes.append(ep)
                    active_episodes.remove(ep)
                elif ep['status'] == 'waiting_final':
                    # 状态2：末尾表达式，移到waiting_final列表，不参与本轮同步
                    waiting_final.append(ep)
                    active_episodes.remove(ep)

            # 阶段3: 检查是否所有剩余活跃episodes都到达状态3同步点
            if active_episodes and all(ep['status'] == 'waiting_intermediate' for ep in active_episodes):
                # 阶段4: 批处理状态3（中间表达式）
                intermediate_expressions = []
                for ep in active_episodes:
                    try:
                        expr = ep['env'].get_current_expression()
                        if expr:
                            intermediate_expressions.append((ep, expr))
                    except:
                        continue

                if intermediate_expressions:
                    # 批量计算中间IC
                    ic_results = self.batch_calculate_intermediate_ic(
                        [expr for ep, expr in intermediate_expressions]
                    )

                    # 分配结果并恢复episodes到运行状态
                    for (ep, expr), ic in zip(intermediate_expressions, ic_results):
                        ep['env'].receive_ic_result(ic)
                        ep['status'] = 'running'

                        # 补全对应的pending transitions
                        self._complete_pending_transitions(ep, ic, 'waiting_intermediate')

            # 阶段5: 处理已完成的末尾表达式episodes
            # 注意：状态2的episodes由另一套异步系统处理，不在这里立即处理

            # 更新活跃episodes列表（只有状态为'running'的）
            active_episodes = [ep for ep in active_episodes if ep['status'] == 'running']

            # 阶段6: 检查是否需要处理待定的状态2表达式
            if len(self.pending_final_expressions) >= self.batch_final_threshold:
                self._process_pending_final_expressions()

            # 处理已完成的episodes
            for episode in completed_episodes:
                self.episodes += 1
                self.train_return.append(episode['return'])

                if self.episodes % self.log_interval == 0:
                    try:
                        self.writer.add_scalar(
                            'ic/train', self.env.env.pool.state['best_ic_ret'], self.steps)
                    except Exception as e:
                        print(f"Warning: Failed to log training stats: {e}")

        # 批次训练间隔处理
        self.train_step_interval()

    def explore_for_env(self, env):
        """为特定环境选择随机动作"""
        allowed_action = env.action_masks()

        if hasattr(env, 'sep_action') and env.sep_action is not None:
            sep_idx = env.sep_action
            if allowed_action[sep_idx]:
                if np.random.rand() < 0.3:
                    return sep_idx

        action = env.action_space.sample()
        while not allowed_action[action]:
            action = env.action_space.sample()
        return action

    def exploit_for_env(self, env, state):
        """为特定环境选择最优动作"""
        state = torch.ByteTensor(state).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            q_values = self.online_net.calculate_q(states=state)
            forbid_action = torch.BoolTensor(~env.action_masks()).to(self.device)
            q_values[:, forbid_action] = -1e6
            action = q_values.argmax().item()
        return action

    def batch_calculate_intermediate_ic(self, expressions):
        """批量计算中间表达式的IC"""
        print(f"🔄 Batch calculating IC for {len(expressions)} intermediate expressions")

        ic_results = []
        for expr in expressions:
            try:
                # 这里调用单个IC计算，实际实现中应该批量化
                ic = self.env.env.pool.calculate_single_ic_for_expr(expr)
                ic_results.append(ic)
            except Exception as e:
                print(f"❌ Failed to calculate IC for intermediate expression: {e}")
                ic_results.append(0.0)

        return ic_results

    def _process_pending_final_expressions(self):
        """处理待定的状态2表达式批处理"""
        if not self.pending_final_expressions:
            return

        print(f"🔄 Processing {len(self.pending_final_expressions)} pending final expressions")

        # 提取表达式列表
        expressions = [expr for ep, expr in self.pending_final_expressions]

        # 批量计算因子值（异步，避免每个表达式都调用Lorentz）
        try:
            print(f"🔬 Starting batch Lorentz computation for {len(expressions)} expressions...")

            # 调用批量因子计算API
            from external_compute import compute_batch_factor_values
            results = compute_batch_factor_values(expressions)

            # 验证所有表达式都已计算完成
            computed_count = 0
            for expr in expressions:
                expr_str = str(expr)
                if expr_str in results and results[expr_str][0] is not None:
                    computed_count += 1
                else:
                    print(f"⚠️ Expression not computed: {expr_str}")

            print(f"✅ Batch computation completed: {computed_count}/{len(expressions)} expressions successful")

            if computed_count == 0:
                raise RuntimeError("All expressions failed to compute")

        except Exception as e:
            print(f"❌ Batch factor computation failed: {str(e)}")
            print("📋 Failed expressions:")
            for i, expr in enumerate(expressions, 1):
                print(f"   {i}. {str(expr)}")
            # 重新抛出异常，让上层处理
            raise

        # 批量交给四池系统处理
        self.batch_calculate_final_ic(expressions)

        # 清空待处理队列
        self.pending_final_expressions.clear()
        print(f"✅ Processed {len(expressions)} final expressions")

    def batch_calculate_final_ic(self, expressions):
        """批量计算末尾表达式的IC（交给四池系统）"""
        print(f"🔄 Batch calculating final IC for {len(expressions)} expressions")

        for expr in expressions:
            try:
                # 交给现有的pool.try_new_expr处理（四池系统）
                # 注意：这里的reward是四池系统计算的延迟reward
                reward = self.env.env.pool.try_new_expr(expr)

                # 找到对应的episode并补全其pending transitions
                # 这里简化处理，假设每个表达式对应一个episode
                # 实际上应该通过更精确的匹配机制
                for ep, ep_expr in self.pending_final_expressions:
                    if str(ep_expr) == str(expr):
                        self._complete_pending_transitions(ep, reward, 'waiting_final')
                        break

            except Exception as e:
                print(f"❌ Failed to process final expression: {e}")

    def train_step_interval(self):
        """批次训练间隔处理"""
        self.epsilon_train.step()

        if self.steps % self.target_update_interval == 0:
            self.update_target()

        if self.is_update():
            self.learn()

        if (self.steps % self.eval_interval == 0) and (len(self.env.env.pool.state['exprs']) >= 1):
            self.evaluate()
            self.save_models(os.path.join(self.model_dir, 'final'))

