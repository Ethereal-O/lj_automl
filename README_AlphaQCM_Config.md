# AlphaQCM 配置系统使用指南

## 📋 概述

AlphaQCM 现在支持通过 `alphaqcm_config.yaml` 配置文件管理系统中的所有超参数。您只需要编辑这个 YAML 文件即可修改系统行为，无需修改代码。

## 🚀 快速开始

### 1. 配置文件位置
```
alphaqcm_config.yaml  # 主配置文件
```

### 2. 修改配置
直接编辑 `alphaqcm_config.yaml` 文件，修改您想要的参数。

### 3. 运行系统
```bash
# 使用配置文件中的参数运行
python train_qcm.py

# 仍然支持命令行覆盖
python train_qcm.py --model fqf --pool 50
```

## ⚙️ 配置参数详解

### RL训练配置 (training)
```yaml
training:
  model: "iqn"          # 模型选择: qrdqn, iqn, fqf
  seed: 0              # 随机种子
  pool_capacity: 30    # 因子池容量 (推荐: 20-50)
  std_lam: 1.0         # 标准差参数
```

### AlphaPool 配置 (alpha_pool)
```yaml
alpha_pool:
  enable_culling: false          # 是否启用因子池淘汰
  culling_method: "ic_drop"      # 淘汰方法: ic_drop, weight, combined
  use_lgb_evaluation: false      # 是否使用LightGBM评估
  reeval_cycle: 1000             # 重新评估周期 (个因子)
  reeval_q5_threshold: 0.5       # q5提升阈值 (bps)
```

### 环境配置 (environment)
```yaml
environment:
  print_expr: true      # 是否打印生成的表达式
  intermediate_weight: 0.3  # 中间奖励权重
  final_weight: 1.0     # 最终奖励权重
```

### 数据配置 (data)
```yaml
data:
  start_date: "20200101"        # 开始日期
  end_date: "20241231"          # 结束日期
  returns_data_root: "./returns" # 收益率数据路径
  factor_cache_dir: "factor_cache"      # 因子缓存目录
  return_cache_dir: "return_cache"      # 返回率缓存目录
  frequency_config: "1dper1d"    # 时间频率配置
```

### Lorentz 配置 (lorentz)
```yaml
lorentz:
  executable_path: "/dfs/dataset/365-1734663142170/data/Lorentz_History-Insider"
  thread_num: 8
  # 其他路径会自动基于frequency_config生成
```

### 系统路径配置 (paths)
```yaml
paths:
  alphaqcm_data_dir: "AlphaQCM_data"
  logs_dir: "alpha_logs"
  factors_output_dir: "/dfs/data/Factors"
  abnormal_stats_dir: "/dfs/data/AbnormalStats"
```

## 🎯 参数优先级

1. **命令行参数** (最高优先级)
   ```bash
   python train_qcm.py --model fqf --pool 50
   ```

2. **配置文件参数** (中等优先级)
   ```yaml
   # alphaqcm_config.yaml
   training:
     model: "iqn"
     pool_capacity: 30
   ```

3. **默认值** (最低优先级)

## 📝 配置示例

### 分钟级高频因子挖掘配置
```yaml
# 高频交易配置
data:
  frequency_config: "5per5"  # 5分钟频率
  start_date: "20240101"
  end_date: "20241231"

alpha_pool:
  pool_capacity: 50  # 更大的池子
  use_lgb_evaluation: true  # 使用完整评估
  reeval_cycle: 500  # 更频繁的重新评估

training:
  model: "iqn"  # 使用IQN算法
  pool_capacity: 50
```

### 日频因子挖掘配置
```yaml
# 日频因子配置
data:
  frequency_config: "1dper1d"  # 日频数据
  start_date: "20180101"
  end_date: "20231231"

alpha_pool:
  pool_capacity: 30
  use_lgb_evaluation: false  # 使用快速评估
  reeval_cycle: 2000  # 较少重新评估

training:
  model: "qrdqn"
  pool_capacity: 30
```

## 🔧 高级配置

### 自定义路径
```yaml
lorentz:
  executable_path: "/custom/path/to/lorentz"
  thread_num: 16

data:
  returns_data_root: "/custom/returns/path"
  factor_cache_dir: "/custom/cache"
```

### 性能优化
```yaml
cache:
  factor_cache_ttl_hours: 48    # 更长的缓存时间
  max_cache_size_mb: 2000       # 更大的缓存空间

debug:
  enable_profiling: true        # 启用性能分析
  save_intermediate_results: true  # 保存中间结果
```

## 🚨 注意事项

### 1. 配置文件格式
- 使用标准的 YAML 格式
- 注意缩进（使用2个空格）
- 字符串值需要用引号

### 2. 参数验证
系统会在启动时验证配置文件的完整性，如果缺少必需参数会报错。

### 3. 热重载
修改配置文件后需要重启程序，系统不支持运行时的热重载。

### 4. 路径配置
- 相对路径是相对于项目根目录
- 绝对路径需要确保在目标系统上存在
- Windows 和 Linux 路径分隔符会自动处理

## 🐛 故障排除

### 配置加载失败
```
FileNotFoundError: 配置文件不存在: alphaqcm_config.yaml
```
**解决**: 确保 `alphaqcm_config.yaml` 文件在项目根目录

### 配置验证失败
```
ValueError: 配置文件缺少必需的配置节: training
```
**解决**: 检查配置文件是否包含所有必需的配置节

### 参数类型错误
```
TypeError: 期望 int 类型，但得到 str
```
**解决**: 检查 YAML 文件中的参数类型是否正确

## 📞 获取帮助

如果您在配置过程中遇到问题，请：

1. 检查 `alphaqcm_config.yaml` 文件的语法
2. 确认参数值在有效范围内
3. 查看控制台错误信息
4. 参考本文档的示例配置

## 🎉 配置优势

通过这个配置文件系统，您可以：

- ✅ **无需编程**: 直接编辑文本文件修改参数
- ✅ **版本控制**: 配置文件可以纳入版本控制
- ✅ **环境隔离**: 不同环境使用不同配置文件
- ✅ **参数共享**: 团队成员共享配置参数
- ✅ **实验记录**: 保存不同实验的配置快照

现在您可以轻松调整 AlphaQCM 的所有参数，进行各种量化因子挖掘实验了！🚀
