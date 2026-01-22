import os
import sys
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
from io import StringIO

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent, BatchAgent
from alphagen_qlib.calculator import ExternalCalculator
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from adapters.scoring_calculator import factor_cache, target_manager
from config_loader import load_config_for_train_qcm, get_alpha_pool_config, get_environment_config, get_paths_config


def external_compute_factor(expr_str):
    """
    计算因子值，支持分钟级别数据聚合
    预热阶段：跳过真实计算，返回mock数据

    Args:
        expr_str: 因子表达式

    Returns:
        values: (n_dates, n_stocks) 因子值数组
        dates: 日期索引
        symbols: 股票代码索引
    """
    try:
        # 检查缓存中是否已有该表达式的因子值
        if factor_cache.has_factor(expr_str):
            cached_data = factor_cache.load_factor(expr_str)
            values = cached_data['values']
            dates = cached_data['dates']
            symbols = cached_data['symbols']
            return values, dates, symbols

        # 调用外部计算脚本 (现在返回分钟级别数据)
        print(f"🔄 Computing factor for: {expr_str}", file=sys.stderr)
        result = subprocess.run(['python3', 'external_compute.py', expr_str], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Lorentz calculation failed for: {expr_str}", file=sys.stderr)
            print(f"Return code: {result.returncode}", file=sys.stderr)
            print(f"Stdout: {result.stdout[:500]}...", file=sys.stderr)
            print(f"Stderr: {result.stderr[:500]}...", file=sys.stderr)
            # 返回零数组作为fallback，但不缓存
            return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])

        # 检查输出是否为空
        if not result.stdout.strip():
            print(f"❌ Lorentz returned empty output for: {expr_str}", file=sys.stderr)
            return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])

        # 解析external_compute.py的输出 (现在包含minuteCode)
        try:
            df = pd.read_csv(StringIO(result.stdout))
        except Exception as parse_error:
            print(f"❌ Failed to parse CSV output for: {expr_str}", file=sys.stderr)
            print(f"Parse error: {parse_error}", file=sys.stderr)
            print(f"Raw output (first 500 chars): {result.stdout[:500]}", file=sys.stderr)
            return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])

        if df.empty:
            print(f"❌ Lorentz returned empty DataFrame for: {expr_str}", file=sys.stderr)
            return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])

        # 检查是否包含minuteCode列（新格式）
        if 'minuteCode' in df.columns:
            # 新格式：按日期聚合，使用最新的分钟数据
            df['date'] = pd.to_datetime(df['date'])

            # 对每只股票的每个日期，选择最新的分钟数据
            # 按日期+股票分组，选择minuteCode最大的记录
            df_latest = df.sort_values(['date', 'symbol', 'minuteCode']).groupby(['date', 'symbol']).last().reset_index()

            # 透视表：行=日期，列=股票代码，值=因子值
            pivot = df_latest.pivot(index='date', columns='symbol', values='factor_value').fillna(0.0)

        else:
            # 旧格式：直接透视
            df['date'] = pd.to_datetime(df['date'])
            pivot = df.pivot(index='date', columns='symbol', values='value').fillna(0.0)

        values = pivot.values
        dates = pivot.index
        symbols = pivot.columns

        # 验证数据质量
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"❌ Invalid factor data for: {expr_str} - all NaN or empty", file=sys.stderr)
            return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])

        print(f"📊 Parsed factor data: shape {values.shape}, saving to cache...", file=sys.stderr)

        # 缓存计算结果
        try:
            factor_cache.save_factor(expr_str, values, dates, symbols)
            print(f"✅ Successfully cached factor: {expr_str}", file=sys.stderr)
        except Exception as cache_error:
            print(f"❌ Failed to cache factor: {expr_str} - {cache_error}", file=sys.stderr)
            # 仍然返回数据，但不缓存
            return values, dates, symbols

        return values, dates, symbols

    except Exception as e:
        print(f"💥 Critical error in external_compute_factor for: {expr_str}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # 返回零数组作为fallback
        return np.zeros((100, 50)), pd.date_range('2020-01-01', periods=100), pd.Index([f'stock_{i}' for i in range(50)])


def run(args):
    # 加载AlphaQCM配置
    alphaqcm_config = load_config_for_train_qcm(args)

    # 获取各个配置节
    alpha_pool_config = get_alpha_pool_config()
    environment_config = get_environment_config()
    paths_config = get_paths_config()

    # 加载RL算法配置
    config_path = os.path.join('qcm_config', f'{args.model}.yaml')
    with open(config_path) as f:
        rl_config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda_available = torch.cuda.is_available()

    # 静默初始化，减少刷屏

    # 使用配置文件中的日期范围
    data_config = alphaqcm_config.get_data_config()
    start_date = data_config.get('start_date', '20200101')
    end_date = data_config.get('end_date', '20241231')

    # 加载真实的target data（用于正式训练阶段的IC计算）
    from adapters.scoring_calculator import load_target_from_csv

    target_data = load_target_from_csv(start_date, end_date)
    if target_data is not None:
        target_manager.save_target(
            target_data['values'],
            target_data['dates'],
            target_data['symbols']
        )
    else:
        # 如果加载失败，使用最小mock数据作为fallback
        import numpy as np
        import pandas as pd

        n_days = 10
        n_stocks = 50

        np.random.seed(42)
        target_values = np.zeros((n_days, n_stocks))
        target_dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        target_symbols = pd.Index([f'{i:06d}' for i in range(1, n_stocks + 1)])

        target_manager.save_target(target_values, target_dates, target_symbols)

    # Use ExternalCalculator with external function (no need for target expression)
    train_calculator = ExternalCalculator(device, external_compute_factor)
    valid_calculator = ExternalCalculator(device, external_compute_factor)
    test_calculator = ExternalCalculator(device, external_compute_factor)

    # Store calculator references
    calculator_refs = {
        'train': train_calculator,
        'valid': valid_calculator,
        'test': test_calculator
    }

    # 使用配置文件创建AlphaPool
    train_pool = AlphaPool(
        capacity=args.pool,
        calculator=train_calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        enable_culling=alpha_pool_config.get('enable_culling', False),
        culling_method=alpha_pool_config.get('culling_method', 'ic_drop'),
        baseline_expressions=[],  # 训练过程中积累的baseline因子
        use_lgb_evaluation=alpha_pool_config.get('use_lgb_evaluation', False),
        reeval_cycle=alpha_pool_config.get('reeval_cycle', 1000),
        reeval_q5_threshold=alpha_pool_config.get('reeval_q5_threshold', 0.5)
    )

    # 使用配置文件创建AlphaEnv
    train_env = AlphaEnv(
        pool=train_pool,
        device=device,
        print_expr=environment_config.get('print_expr', True),
        intermediate_weight=environment_config.get('intermediate_weight', 0.3),
        final_weight=environment_config.get('final_weight', 1.0)
    )

    # Specify the directory to log.
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")

    # 使用配置文件中的路径
    alphaqcm_data_dir = paths_config.get('alphaqcm_data_dir', 'AlphaQCM_data')
    logs_dir = paths_config.get('logs_dir', 'alpha_logs')

    if name in ['qrdqn', 'iqn']:
        log_dir = os.path.join(alphaqcm_data_dir, logs_dir,
                           f'pool_{args.pool}_QCM_{args.std_lam}',
                           f"{name}-seed{args.seed}-{time}-N{rl_config['N']}-lr{rl_config['lr']}-per{rl_config['use_per']}-gamma{rl_config['gamma']}-step{rl_config['multi_step']}")
    elif name == 'fqf':
        log_dir = os.path.join(alphaqcm_data_dir, logs_dir,
                           f'pool_{args.pool}_QCM_{args.std_lam}',
                           f"{name}-seed{args.seed}-{time}-N{rl_config['N']}-lr{rl_config['quantile_lr']}-per{rl_config['use_per']}-gamma{rl_config['gamma']}-step{rl_config['multi_step']}")

    # Filter out data configuration parameters that shouldn't be passed to agent
    agent_config = {k: v for k, v in rl_config.items()
                   if k not in ['START_DATE', 'END_DATE']}

    # Create the agent and run.
    if args.batch_mode:
        if name == 'qrdqn':
            agent = BatchAgent(
                env_template=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                num_parallel_envs=args.num_envs,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
        elif name == 'iqn':
            agent = BatchAgent(
                env_template=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                num_parallel_envs=args.num_envs,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
        elif name == 'fqf':
            agent = BatchAgent(
                env_template=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                num_parallel_envs=args.num_envs,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
    else:
        if name == 'qrdqn':
            agent = QRQCMAgent(
                env=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
        elif name == 'iqn':
            agent = IQCMAgent(
                env=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
        elif name == 'fqf':
            agent = FQCMAgent(
                env=train_env,
                valid_calculator=valid_calculator,
                test_calculator=test_calculator,
                log_dir=log_dir,
                seed=args.seed,
                std_lam=args.std_lam,
                cuda=cuda_available,
                **agent_config
            )
    # Set agent reference for Q calculation
    train_env.env.agent = agent
    # Set agent references in calculators for warmup phase detection
    def weak_agent_ref():
        return agent  # 直接返回外部作用域的agent变量

    for calc_name, calc in calculator_refs.items():
        if hasattr(calc, '_agent_ref'):
            calc._agent_ref = weak_agent_ref
    try:
        agent.run()
    except Exception as e:
        print(f"Training error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

        try:
            agent.save_models(os.path.join(agent.model_dir, 'final'))
        except Exception as save_error:
            print(f"Failed to save models: {save_error}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool', type=int, default=20)
    parser.add_argument('--std-lam', type=float, default=1.0)
    parser.add_argument('--batch-mode', action='store_true',
                       help='Enable batch processing mode with multiple parallel episodes')
    parser.add_argument('--num-envs', type=int, default=4,
                       help='Number of parallel environments in batch mode')
    args = parser.parse_args()
    run(args)
