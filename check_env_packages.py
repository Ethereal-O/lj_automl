#!/usr/bin/env python3
"""
检查 alphaqcm_env.yml 中包的安装状态
在服务器上运行此脚本，查看哪些包还没安装

使用方法:
python check_env_packages.py
"""

import yaml
import importlib
import sys
import os

def main():
    """主函数"""
    print('🔍 检查 alphaqcm_env.yml 中的包安装状态')
    print('=' * 60)

    # 检查环境文件是否存在
    env_file = 'alphaqcm_env.yml'
    if not os.path.exists(env_file):
        print(f'❌ 找不到环境文件: {env_file}')
        print('请确保 alphaqcm_env.yml 文件在当前目录')
        return 1

    # 读取环境文件
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            env_data = yaml.safe_load(f)
    except Exception as e:
        print(f'❌ 读取环境文件失败: {e}')
        return 1

    # 提取pip包
    pip_packages = []
    if 'dependencies' in env_data:
        for dep in env_data['dependencies']:
            if isinstance(dep, dict) and 'pip' in dep:
                pip_packages = dep['pip']
                break

    if not pip_packages:
        print('❌ 在环境文件中找不到pip依赖')
        return 1

    print(f'📦 总共需要检查 {len(pip_packages)} 个包')
    print()

    missing_packages = []
    version_mismatches = []
    installed_correctly = []

    for pkg_spec in pip_packages:
        try:
            # 解析包名和版本
            if '==' in pkg_spec:
                pkg_name, required_version = pkg_spec.split('==', 1)
            else:
                pkg_name = pkg_spec
                required_version = None

            # 尝试导入包
            try:
                # 处理特殊包名映射
                import_name = pkg_name.replace('-', '_')
                if pkg_name == 'scikit-learn':
                    import_name = 'sklearn'
                elif pkg_name == 'pyyaml':
                    import_name = 'yaml'

                module = importlib.import_module(import_name)

                # 获取已安装版本
                try:
                    installed_version = getattr(module, '__version__', 'unknown')
                except:
                    installed_version = 'unknown'

                if required_version:
                    if installed_version == required_version:
                        installed_correctly.append(f'{pkg_name}=={installed_version}')
                    else:
                        version_mismatches.append(f'{pkg_name} (需要: {required_version}, 已安装: {installed_version})')
                else:
                    installed_correctly.append(f'{pkg_name}=={installed_version}')

            except ImportError:
                missing_packages.append(pkg_spec)

        except Exception as e:
            print(f'❓ 检查 {pkg_spec} 时出错: {e}')
            missing_packages.append(pkg_spec)

    print('✅ 已正确安装:')
    for pkg in installed_correctly[:5]:  # 只显示前5个
        print(f'  {pkg}')
    if len(installed_correctly) > 5:
        print(f'  ... 还有 {len(installed_correctly) - 5} 个包已正确安装')

    print()
    print('⚠️  版本不匹配:')
    for pkg in version_mismatches[:5]:  # 只显示前5个
        print(f'  {pkg}')
    if len(version_mismatches) > 5:
        print(f'  ... 还有 {len(version_mismatches) - 5} 个包版本不匹配')

    print()
    print('❌ 未安装的包:')
    for pkg in missing_packages[:10]:  # 只显示前10个
        print(f'  {pkg}')
    if len(missing_packages) > 10:
        print(f'  ... 还有 {len(missing_packages) - 10} 个包未安装')

    print()
    print('📊 总结:')
    print(f'  ✅ 正确安装: {len(installed_correctly)} 个')
    print(f'  ⚠️  版本不匹配: {len(version_mismatches)} 个')
    print(f'  ❌ 未安装: {len(missing_packages)} 个')

    if missing_packages or version_mismatches:
        print()
        print('🔧 需要安装的包 (复制给IT):')

        # 版本不匹配的包也需要重新安装
        all_needed = missing_packages[:]
        for mismatch in version_mismatches:
            pkg_name = mismatch.split(' (')[0]
            required_ver = mismatch.split('需要: ')[1].split(',')[0]
            all_needed.append(f'{pkg_name}=={required_ver}')

        for pkg in all_needed:
            print(pkg)

        print()
        print('⚠️  重要: PyTorch需要单独安装CUDA版本')
        print('pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116')

        return 1
    else:
        print('🎉 所有包都已正确安装！')
        print('🚀 可以运行: python train_qcm.py --model iqn --pool 30')
        return 0

if __name__ == "__main__":
    sys.exit(main())
