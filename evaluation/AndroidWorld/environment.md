# AndroidWorld 环境配置（macOS / conda / Python 3.11）

本文档记录一套可复现的 AndroidWorld 环境安装流程。

---

### 1) 创建并激活虚拟环境

```bash
conda create -n android_world python=3.11.8
conda activate android_world
```

---

### 2) 安装 AndroidWorld（requirements + opencv + 安装包）

在 `evaluation/AndroidWorld` 路径下执行：

```bash
cd /Users/cckevin/ScaleCUA/evaluation/AndroidWorld

# 安装 AndroidWorld 官方 requirements
python -m pip install -r requirements.txt

# opencv 单独用 conda-forge 安装（避免 pip 版本带来的兼容问题）
conda install -c conda-forge opencv

# 安装 AndroidWorld 包（会触发 proto 代码生成）
python setup.py install
```

---

### 3) 安装模型 API 依赖（OpenAI / Anthropic）

```bash
python -m pip install openai
python -m pip install anthropic
```

---

### 4) 安装 android_env（AndroidEnv）

```bash
cd /Users/cckevin/ScaleCUA/evaluation/AndroidWorld

git clone https://github.com/deepmind/android_env/
cd android_env
python -m pip install -e .
```

参考仓库：[google-deepmind/android_env](https://github.com/google-deepmind/android_env)

---

### 5) 固定 Protobuf 版本（解决 gencode/runtime 不一致）

```bash
python -m pip install -U "protobuf==6.31.1"
```

可选验证：

```bash
python -c "import google.protobuf as p; print('protobuf runtime =', p.__version__)"
```

---

### 6) 解决 SQLite 的 FTS4 问题（`no such module: fts4`）

#### a) 清理可能的 sqlite 绑定包

```bash
python -m pip uninstall -y pysqlite3 pysqlite3-binary || true
python -m pip show pysqlite3 pysqlite3-binary || true
```

#### b) 用 conda-forge 装/重装 sqlite（提供可用的 libsqlite + headers）

```bash
conda install -c conda-forge -y --force-reinstall sqlite libsqlite pkg-config
```

#### c) 设置编译参数（启用 FTS3/FTS4）

```bash
export CPPFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
export MACOSX_DEPLOYMENT_TARGET=11.0
export CFLAGS="${CPPFLAGS} -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4"
```

#### d) 强制从源码编译安装 pysqlite3（不要 wheel）

```bash
python -m pip install --no-binary :all: --no-cache-dir pysqlite3
```

#### e) 验证 FTS4 是否可用（必须不报错）

```bash
python -c "import pysqlite3.dbapi2 as s; c=s.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts4(x)'); print('FTS4 OK'); print([x[0] for x in c.execute('pragma compile_options') if 'FTS' in x[0]])"
```

---

### 7) 运行验证（示例）

启动AndroidWorldAvd:
```bash
EMULATOR_NAME=AndroidWorldAvd
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```

回到 AndroidWorld 目录：

```bash
cd /Users/cckevin/ScaleCUA/evaluation/AndroidWorld
python run_random_walk.py
```

