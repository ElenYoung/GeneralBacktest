# 上传到 PyPI 指南

本文档说明如何将 GeneralBacktest 包上传到 PyPI（Python Package Index）。

## 📋 准备工作

### 1. 安装必要的工具

```bash
pip install --upgrade pip
pip install --upgrade build twine
```

### 2. 更新项目信息

在上传之前，请修改以下文件中的占位符信息：

#### pyproject.toml
```toml
[project]
name = "GeneralBacktest"  # 如果 PyPI 上已有同名包，需要改名，如 "GeneralBacktest-YourName"
authors = [
    {name = "Your Name", email = "your.email@example.com"}  # ← 修改为你的信息
]

[project.urls]
"Homepage" = "https://github.com/yourusername/GeneralBacktest"  # ← 修改为你的仓库地址
"Bug Reports" = "https://github.com/yourusername/GeneralBacktest/issues"
"Source" = "https://github.com/yourusername/GeneralBacktest"
```

#### README.md
- 修改作者信息
- 修改 GitHub 链接
- 根据需要调整文档内容

### 3. 检查版本号

在 [pyproject.toml](pyproject.toml) 和 [src/GeneralBacktest/__init__.py](src/GeneralBacktest/__init__.py) 中确认版本号一致：

```python
__version__ = '1.0.0'
```

## 🔨 构建包

在项目根目录执行：

```bash
python -m build
```

这会在 `dist/` 目录下生成两个文件：
- `GeneralBacktest-1.0.0.tar.gz` (源码包)
- `GeneralBacktest-1.0.0-py3-none-any.whl` (wheel 包)

## 🧪 测试上传到 TestPyPI（推荐）

在正式上传前，建议先上传到测试服务器验证。

### 1. 注册 TestPyPI 账号

访问 https://test.pypi.org/account/register/ 注册账号

### 2. 生成 API Token

1. 登录 TestPyPI
2. 访问 https://test.pypi.org/manage/account/token/
3. 创建新的 API token
4. 保存 token（格式：`pypi-xxx...`）

### 3. 上传到 TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

输入用户名：`__token__`
输入密码：你的 API token

### 4. 测试安装

```bash
pip install --index-url https://test.pypi.org/simple/ GeneralBacktest
```

如果有依赖问题，可以混合使用：

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple GeneralBacktest
```

## 🚀 正式上传到 PyPI

### 1. 注册 PyPI 账号

访问 https://pypi.org/account/register/ 注册账号

### 2. 生成 API Token

1. 登录 PyPI
2. 访问 https://pypi.org/manage/account/token/
3. 创建新的 API token
4. 保存 token

### 3. 上传到 PyPI

```bash
python -m twine upload dist/*
```

输入用户名：`__token__`
输入密码：你的 API token

### 4. 验证安装

```bash
pip install GeneralBacktest
```

## 🔄 更新包

当需要发布新版本时：

1. **更新版本号**：修改 `pyproject.toml` 和 `src/GeneralBacktest/__init__.py` 中的版本号

2. **清理旧构建**：
   ```bash
   Remove-Item -Recurse -Force dist, build, *.egg-info
   ```

3. **重新构建**：
   ```bash
   python -m build
   ```

4. **上传新版本**：
   ```bash
   python -m twine upload dist/*
   ```

## 📁 最终的项目结构

上传前，确保项目结构如下：

```
GeneralBacktest/
├── .gitignore                    # Git 忽略文件
├── LICENSE                       # MIT 许可证
├── MANIFEST.in                   # 包含额外文件的清单
├── PYPI_UPLOAD_GUIDE.md         # 本文档
├── README.md                     # 项目说明（PyPI 展示）
├── pyproject.toml               # 现代 Python 打包配置（主要）
├── setup.py                      # 向后兼容文件
├── examples/                     # 示例代码（可选）
├── src/
│   └── GeneralBacktest/
│       ├── __init__.py          # 包初始化文件
│       ├── backtest.py          # 主要回测类
│       └── utils.py             # 工具函数
└── output_demo/                 # 示例输出（可选）
```

## ⚠️ 重要注意事项

### 包名冲突
如果 PyPI 上已存在 `GeneralBacktest` 包，你需要：
1. 修改包名，如 `GeneralBacktest-YourName`
2. 更新 `pyproject.toml` 中的 `name` 字段
3. 告知用户使用新的包名安装

### 敏感信息
- **不要**在代码中包含数据库配置文件（已在 `.gitignore` 中排除）
- **不要**提交 API keys 或密码
- `run_backtest_ETF()` 和 `run_backtest_stock()` 方法需要用户自行配置数据库

### 文档说明
在 README 中已明确说明：
- 普通用户使用 `run_backtest()` 方法
- `run_backtest_ETF()` 和 `run_backtest_stock()` 需要特殊的数据库配置
- 这些方法主要为特定用户群体（课程学员）提供

## 🔧 故障排除

### 问题：上传时提示包名已存在
**解决**：修改 `pyproject.toml` 中的包名，添加后缀使其唯一

### 问题：依赖安装失败
**解决**：检查 `pyproject.toml` 中的依赖版本是否正确

### 问题：导入失败
**解决**：
1. 确认 `src/GeneralBacktest/__init__.py` 正确导出了类
2. 检查相对导入路径

### 问题：README 在 PyPI 上显示不正常
**解决**：确保 README.md 使用标准 Markdown 格式，避免使用 GitHub 特有的扩展语法

## 📚 参考资源

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI 帮助文档](https://pypi.org/help/)
- [Twine 文档](https://twine.readthedocs.io/)
- [PEP 517 - 构建后端接口](https://peps.python.org/pep-0517/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)

## ✅ 上传检查清单

上传前请确认：

- [ ] 修改了 `pyproject.toml` 中的作者信息和项目链接
- [ ] 修改了 `README.md` 中的作者信息
- [ ] 版本号在 `pyproject.toml` 和 `__init__.py` 中一致
- [ ] 运行 `python -m build` 成功构建包
- [ ] 先在 TestPyPI 上测试上传和安装
- [ ] LICENSE 文件存在且正确
- [ ] README.md 内容完整且格式正确
- [ ] 没有包含敏感信息（数据库配置等）
- [ ] `.gitignore` 正确配置
- [ ] 代码通过基本测试

祝你上传成功！🎉
