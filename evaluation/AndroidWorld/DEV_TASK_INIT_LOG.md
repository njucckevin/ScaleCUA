# AndroidWorld 任务初始化与数据复现：开发日志

> 目的：记录本次排查中关于 AndroidWorld `TaskEval.initialize_task(env)`、seed/params、随机噪声数据复现、以及数据集过滤策略的关键结论，供后续开发/接任者快速理解与继续推进。

---

## 1. AndroidWorld “任务初始化”到底做了什么

### 1.1 统一入口：`suite_utils._run_task()` 调 `task.initialize_task(env)`

调用发生在跑 episode 之前：

```233:241:evaluation/AndroidWorld/android_world/suite_utils.py
task.initialize_task(env)
interaction_results = run_episode(task)
```

### 1.2 通用初始化逻辑：`TaskEval.initialize_task()`

关键动作：
- **清理交互缓存**：`env.interaction_cache = ""`
- **固定设备时间**：默认设为 `device_constants.DT`（除少数任务覆写）
- **恢复 app 私有数据快照**（针对 `app_names`）：`app_snapshot.restore_snapshot(app_name, env.controller)`
- **设置随机种子**：如果 `params["seed"]` 存在 → `random.seed(seed)`

代码参考：

```132:150:evaluation/AndroidWorld/android_world/task_evals/task_eval.py
env.interaction_cache = ""
self.initialize_device_time(env)
self._initialize_apps(env)
...
seed = self.params.get("seed")
if seed is not None:
    random.seed(seed)
```

### 1.3 app 私有数据快照恢复：`app_snapshot.restore_snapshot()`

对 `/data/data/<package>` 进行强覆盖恢复：
- `close_app`
- 清空 app data 目录
- 从 snapshot 目录拷回
- `restorecon` / `chmod 777` 修复 SELinux/权限

```75:120:evaluation/AndroidWorld/android_world/utils/app_snapshot.py
adb_utils.close_app(app_name, env)
file_utils.clear_directory(app_data_path, env)
file_utils.copy_dir(snapshot_path, app_data_path, env)
...
```

> 结论：**只要 snapshot 本身不变，`/data/data/...` 私有数据基线可强一致**。

---

## 2. `run.py` vs `run_diy.py`：seed/instance_id 是否一致

### 2.1 `run.py` 的实例化路径
`run.py` 使用 `suite_utils.create_suite(..., seed=_TASK_RANDOM_SEED, use_identical_params=_FIXED_TASK_SEED)`：

```286:292:evaluation/AndroidWorld/run.py
suite = suite_utils.create_suite(..., seed=_TASK_RANDOM_SEED.value, use_identical_params=_FIXED_TASK_SEED.value)
```

内部：
- instance seed = `sha256(f"{seed}_{task_name}_{instance_id}") % 2**32`
- `random.seed(instance_seed)` 后 `generate_random_params()`
- 写入 `params["seed"] = instance_seed`

```70:95:evaluation/AndroidWorld/android_world/suite_utils.py
if seed is not None:
    random.seed(seed)
params = task.generate_random_params()
params[constants.EpisodeConstants.SEED] = seed
```

### 2.2 `run_diy.py`（DIY runner）复刻了同样的 seed 派生（重点：对齐 `task_random_seed` + `instance_id`）
DIY 的关键是确保三元组一致：
- `base_task_name`
- `task_random_seed`
- `instance_id` 选择策略（是否固定为 0）

**关键坑**：
- `run.py` 里 `_FIXED_TASK_SEED` 默认 False（原版逻辑），`run_diy.py` 曾设 True；若 `n_task_combinations>1` 且两边 fixed 不一致，会导致 seed/params 不一致。

---

## 3. “params 复现”与“随机噪声数据复现”的本质区别

### 3.1 `params` 是什么
- `params` 是 `dict[str, Any]`，由任务的 `schema` 校验。
- 框架/runner 会写入 `params["seed"]`（`constants.EpisodeConstants.SEED`）。

```41:45:evaluation/AndroidWorld/android_world/task_evals/task_eval.py
jsonschema.validate(params, self.schema)
```

## 3.2 仅保存 params（不含 seed）时，为什么“截图里有、复跑时没了”（详细复盘）

这一节是本次排查的核心：**你当时的截图/随机游走看到的“环境数据”，并不等价于 `generate_random_params()` 返回的 params**。尤其当任务初始化会额外生成“噪声数据”时，`params` 不含 `seed` 将导致截图里的很多数据点无法稳定复现。

下面按我们讨论/定位的真实过程复盘。

### 3.2.1 你当时的采样流程（random walk）缺了哪一步

你当时在 `evaluation/random_walk_aw.py` 里做的是：
- 直接 `params = task_type.generate_random_params()`
- `pickle.dump(params, ...)`
- `task = task_type(params); task.initialize_task(env)`

关键片段：

```374:378:evaluation/random_walk_aw.py
params = task_type.generate_random_params()
with open(os.path.join(PARAMS_DIR, task_uuid+\"_params.pkl\"), \"wb\") as f:
    pickle.dump(params, f)
```

这段流程**没有**像 `suite_utils` 那样把 seed 注入 params：

```94:95:evaluation/AndroidWorld/android_world/suite_utils.py
params[constants.EpisodeConstants.SEED] = seed
```

它带来的直接后果是：
- 采样时 `random` 的状态完全取决于“当时进程运行到哪里、此前消耗了多少次 random”
- 同一个 task 模板每次初始化得到的“噪声数据集”会变（且你没有保存能复原它的关键信息）

### 3.2.2 “截图里有、复跑时没了”的本质：你依赖的目标是 **noise** 而不是 **params-目标**

我们发现两类“从截图联想出来的目标”特别容易踩坑：
- **(A) 目标来自初始化生成的噪声条目**（noise rows / noise files / noise activities）
- **(B) 目标来自 UI 的截断显示**（`.../…`），本身不是稳定标识

当你复跑时，如果没有把噪声生成过程固定下来（seed/随机状态一致），这些目标就会消失或变形。

### 3.2.3 关键证据 1：OpenTracks 的 “Long Distance Run” 消失（IR/动态任务）

你提供的 case：IR 任务 `SportsTrackerTotalDistanceForCategoryOverInterval`。

你当时从截图里看到 OpenTracks 列表里有 `Long Distance Run`，复跑时找不到；打印 pkl params 发现只包含像下面这样的字段：
- `start_date/end_date/category/duration/distance/start_time/elevation/...`
- `activity_name='Recovery day'` 等

关键理解：**IR 的 params 只描述 prompt 槽位，并不描述“初始化后完整数据集”**。

IR/OpenTracks 初始化里明确会额外生成 20 条随机活动作为噪声：

```44:55:evaluation/AndroidWorld/android_world/task_evals/information_retrieval/activity_app_utils.py
activities += _generate_random_activities(20, exclusion_conditions)
```

而 `Long Distance Run` 正是随机候选名之一：

```239:250:evaluation/AndroidWorld/android_world/task_evals/information_retrieval/activity_app_utils.py
\"Long Distance Run\",
```

因此截图里出现 `Long Distance Run` 的真实来源很可能是：
- 不是 `relevant_state`（params 槽位）里的活动
- 而是那 20 条随机噪声活动之一

一旦复跑时随机状态不同（尤其你当时未保存 seed），该噪声集合就会变，`Long Distance Run` 很可能不再出现。

### 3.2.4 关键证据 2：Files 的 `image_file_2023...` 消失（noise 变体文件名）

你提供的 Files case：
- 指令引用了 `image_file_2023...`
- 但加载的 params 形如：
  - `file_name: '0bSM_hot_quilt.exe'`
  - `subfolder: 'Download'`
  - `noise_candidates: ['image_file.png', ...]`

这说明 `image_file_2023...` 并不是任务“稳定目标文件”（那个是 `params[\"file_name\"]`），而更像是噪声文件的**随机变体**。

在 Files 的删除/移动 validator 里，初始化会生成噪声文件：

```130:143:evaluation/AndroidWorld/android_world/task_evals/common_validators/file_validators.py
user_data_generation.generate_noise_files(
    self.params[\"file_name\"],
    self.data_directory,
    env.controller,
    self.params[\"noise_candidates\"],
)
```

而 noise 文件名的变体生成会随机选择“日期前缀/随机后缀/固定后缀”等策略（典型产生 `2023_...` 或 `_AB12` 这类形态）：

```111:132:evaluation/AndroidWorld/android_world/task_evals/utils/user_data_generation.py
modification_type = random.choice([\"date_prefix\", \"random_suffix\", \"fixed_suffix\"])
```

结论：
- `params` 不含 seed 时，你无法稳定复现某个具体 noise 变体文件名
- 你从截图抄到的 `image_file_2023...` 很可能是当时随机噪声之一

### 3.2.5 关键证据 3：Markor 的 `2023_03_23_...` 消失（noise 文件 + UI 截断）

Markor 类似 Files：很多任务初始化会生成 noise 文件名变体，并且 UI 里常以截断 `...` 显示。

因此指令里引用形如：
- `'2023_03_23_insurance_plan_comparis...'`

有两重不稳定：
- `2023_03_23_` 很可能是 noise 变体（依赖随机状态）
- `...` 是 UI 截断，不是稳定标识

这也是我们后来在过滤脚本里增加 “日期前缀 + 引号内 `.../…`” 的原因。

### 3.2.6 为什么“你现在强行写 seed”反而更不一致（一个反直觉点）

你后来在复跑（DIY）时尝试把 seed 写回 params（类似 `params[\"seed\"]=seed`）。

这会把噪声生成固定到“另一套确定序列”，但这套序列与当时 random walk 的“自然随机序列”**几乎必然不同**：
- 当时 random walk：没有 seed，random 状态由进程历史决定（不可追溯）
- 现在 DIY：seed 派生是确定的（`sha256(task_random_seed, task_name, instance_id)`），会产生稳定但不同的 noise 集合

因此：**“补 seed”能让未来可复现，但不能让过去缺 seed 的截图可复现**。

### 3.2.7 可操作结论（面对“旧截图” vs “未来采样”两种目标）

**面对旧截图（random_walk_aw 产物）**：
- 不要依赖 noise 条目/变体文件名/截断文件名作为目标
- 过滤掉指令里出现明显 noise 指征的样本（日期前缀、引号内 `...`、uuid、随机 4 位 token 等）
- 只保留那些“目标数据点直接在 params 中显式存在”的样本（例如 Expense 的 `row_objects/noise_row_objects` 直接保存了完整数据集）

**面对未来采样（想让数据强复现）**：
- 采样时就按 `suite_utils` 思路把 `params[\"seed\"]` 存上（并在生成 params 前 `random.seed(seed)`）
- 对生成噪声的任务，最好额外保存“最终噪声列表”（或直接 dump app DB/目录清单）以实现强一致

### 3.3 典型踩坑案例 1：OpenTracks（IR）里的 “Long Distance Run”

`SportsTrackerTotalDistanceForCategoryOverInterval` 属于 **Information Retrieval 动态任务**：
- `relevant_state` 只定义了少量“相关活动”
- 初始化时还会**额外生成 20 条随机活动（噪声）**

```44:55:evaluation/AndroidWorld/android_world/task_evals/information_retrieval/activity_app_utils.py
activities += _generate_random_activities(20, exclusion_conditions)
```

随机活动名来自候选表，其中包含：

```239:250:evaluation/AndroidWorld/android_world/task_evals/information_retrieval/activity_app_utils.py
"Long Distance Run",
```

seed → “Long Distance Run”的路径：
- `params["seed"]` → `TaskEval.initialize_task()` → `random.seed(seed)`
- `_generate_random_activity()` 连续调用 `random.choice` 选 `category` 与 `name`

### 3.4 典型踩坑案例 2：Files 里的 `image_file_2023...`

`FilesDeleteFile` 的初始化会创建：
- 目标文件：`params["file_name"]`（确定）
- 噪声文件：`generate_noise_files()`（名字是变体随机生成）

因此从截图抄到的 `image_file_2023...` 多半是 **noise 变体**，缺 seed 难以复现。

---

## 4. 为什么有些任务“即使没 seed 也能复现”

如果任务把“相关数据 + 噪声数据”都直接放在 params 里（例如数据库 row_objects/noise_row_objects），复跑时只要复用 params 就可复现。

例：`ExpenseDeleteMultiple2` 的 params 中包含：
- `row_objects`（目标）
- `noise_row_objects`（噪声，包含指令提到的条目）

> 结论：**params 是否“包含完整初始化数据集”决定了无 seed 时能否复现。**

---

## 5. `datetime.now()` 能不能复现

两种场景：
- **只靠 seed**：不能（`datetime.now()` 不受 `random.seed` 控制）。
- **保存 params 结果值并复用**：可以（因为你复用的是 `folder_name` 等结果，而非再次调用 now）。

典型：`MarkorCreateFolder.generate_random_params()` 用 `datetime.now()` 生成 `folder_name`。

---

## 6. pickle params 加载报错的根因与解决

现象：
- 有些 pkl 仅存 `list/dict` → 可直接 `pickle.load`
- 有些 pkl 存了 `android_world.*` 的对象实例 → unpickle 会 import 模块
- 进一步又触发 `android_env` 依赖 → `ModuleNotFoundError: android_env`

解决：
- 需要在相同环境安装 AndroidEnv（`android_env`），按项目文档：
  - `docs/README_AndroidWorld.md` 指引安装 AndroidEnv（clone 并 `python setup.py install`）
  - 官方 repo参考：[`google-research/android_world`](https://github.com/google-research/android_world)

建议：不要在 conda base 装，单独新建 env 更干净。

---

## 7. 数据集过滤：从“能跑起来/更稳定”角度的经验规则

### 7.1 base_task_name 层面的粗过滤（已实践）
先剔除最容易依赖噪声的类别：
- `SportsTracker*`, `Tasks*`, `Notes*`（IR 动态任务）
- `FilesDeleteFile`, `FilesMoveFile`
- `Retro*`
- `SimpleSmsReply*`

### 7.2 instruction 文本层面的 noise 指征过滤（已实现到 `evaluation/AndroidWorld/temp.py`）
在不读取 params 的前提下，用启发式剔除“明显是截图噪声/截断文件名”的样本：
- **日期前缀**：`20YY_MM_DD` / `20YY-MM-DD`
- **引号内出现 `...` / `…`**（截图截断）
- **UUID 形态**：`8-4-4-4-12` hex
- **疑似随机变体文件名**（引号内、且像文件名）：
  - 4 位随机前缀：`0bSM_hot_quilt.exe`
  - 扩展名前 4 位随机 token：`*_AB12.png`

运行结果示例（最后一次）：
- Total 906 | Kept 651 | Dropped 255
- drop reason: base_task_rule 238, noise_filename_in_instruction 17

> 注：这些规则是“高精度低召回”，用于先把 pipeline 跑通。

---

## 8. 对未来采样/合成数据的建议（避免再踩坑）

1) 采样阶段就按 `suite_utils` 思路保存 seed：
   - `random.seed(seed)` → `params = generate_random_params()` → `params["seed"]=seed` → dump
2) 对会生成噪声数据的任务，最好额外保存：
   - “最终生成的 noise 名单”（文件名列表/活动名列表），或
   - 直接 dump app DB/目录清单（强一致）
3) 合成指令时避免引用“截图截断/噪声变体”作为目标；尽量引用 params 中的目标字段。

---

## 9. 相关文件索引（常用）

- `evaluation/AndroidWorld/android_world/task_evals/task_eval.py`：通用初始化、seed 绑定点
- `evaluation/AndroidWorld/android_world/utils/app_snapshot.py`：app 私有数据快照恢复
- `evaluation/AndroidWorld/android_world/suite_utils.py`：suite 创建、seed 派生规则
- `evaluation/AndroidWorld/android_world/task_evals/information_retrieval/activity_app_utils.py`：OpenTracks IR 噪声活动生成（含 “Long Distance Run”）
- `evaluation/AndroidWorld/android_world/task_evals/utils/user_data_generation.py`：noise 文件名变体生成逻辑
- `evaluation/AndroidWorld/temp.py`：当前用于过滤 synthesized_tasks 的脚本（含启发式规则）

