"""
setup.py - 向后兼容的安装配置文件

注意：现代 Python 打包应使用 pyproject.toml
此文件仅为向后兼容旧版本工具（如 pip < 21.3）保留

所有配置都在 pyproject.toml 中定义
"""

from setuptools import setup

# 所有配置在 pyproject.toml 中定义
# 这个文件仅用于兼容旧版本的 pip 和构建工具
setup()
