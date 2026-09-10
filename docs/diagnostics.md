# 诊断输出：日志与 result

> 状态：设计已定，尚未实现。实现落地后删除本行。

两个正交的可选输出，默认都落在输出音频旁边。

## 1. 职责切分

```
日志       →  过程。人读，时间序，逐条 flush，中断即可 tail。
result     →  结果。机读，单个自洽 JSON 文档，与输出音频配套。
segments   →  不变。仍是可编辑、可续跑的输入格式。
```

判据：**能不能在运行中途看** 归日志，**跑完拿去分析** 归 result。

此前只有一条半成品路径：`print` 散落在四个模块（不能关、不能定向、无级别、无时间戳，且污染库调用方的 stdout），`--report` 的逐块 JSONL 只有块级诊断——没有参考侧位置、没有段级信息、没有锚点、没有运行参数，脱离命令行历史就不自洽。

## 2. 文件命名与开关

三个文件默认全开，落在输出音频同目录、同 stem：

```
out.flac  →  out-log.txt
             out-result.json
             out-segments.json
```

`--scan-only` 没有输出音频，回落到 mixture 的目录与 stem。

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `--log PATH` | `<stem>-log.txt` | 改位置 |
| `--no-log` | off | 关闭日志文件（控制台不受影响） |
| `--log-level {debug,info,warning,error}` | `info` | 控制台与文件的共同级别 |
| `--quiet` | off | 仅把控制台钳到 `warning`，文件仍按 `--log-level` |
| `--result PATH` | `<stem>-result.json` | 改位置 |
| `--no-result` | off | 关闭 |
| `--result-anchors {none,summary,full}` | `summary` | 锚点导出档位 |
| `--segments-out PATH` | `<stem>-segments.json` | 改位置 |
| `--no-segments-out` | off | 关闭 |
| `--segments PATH` | 无 | 读入（不变） |

两条规则：

- **给了 `--segments`（读入）时默认不写 segments-out**——段落来自文件，再写一份只是复制。显式给 `--segments-out PATH` 才写。
- 目标文件已存在则覆盖，与输出音频一致。分段续跑时输出音频名本就不同，附属文件随之分开，不会互相覆盖。

于是最常用形态是一行：

```bash
audio-overlap-removal mix.webm ref.webm out.flac
```

屏幕清爽、文件里留全量：

```bash
audio-overlap-removal mix.webm ref.webm out.flac --log-level debug --quiet
```

**为何默认开、而不是 `--log` 无值触发默认名**：`nargs="?"` 与位置参数相邻时会把 `mix.webm` 吃成选项的值，这类歧义无法靠文档规避。默认开 + `--no-*` 零歧义；长任务本来也几乎总该留日志，`summary` 档三个文件通常合计不到 1 MB。

## 3. 日志

### 3.1 库/CLI 边界

库只取 logger，永不配置 handler：

```python
logger = logging.getLogger(__name__)   # audio_overlap_removal.pipeline 等
```

`__init__.py` 给包根 logger 挂 `NullHandler`。配置只发生在 `cli.py`；库使用者（直接调 `process_audio`）自行决定去向。

### 3.2 格式

- 控制台：`%(message)s`，保持现有紧凑外观（`  match C=120.0-3612.4s offset=...`），观感零改动。
- 文件：`%(asctime)s %(levelname)-7s %(name)s %(message)s`，`asctime` 用 UTC ISO-8601（毫秒）。
- 控制台走 **stderr**。此前进度打在 stdout，与任何可管道内容混在一起；挪走后 stdout 干净。

### 3.3 flush

`StreamHandler.emit()` 每条自带 `flush()`，`FileHandler` 继承之，默认即逐条落盘。量级是每 30 秒音频一条，开销可忽略。不引入 buffering、不引入定时器：定时 flush 只会让「中断后看到跑到哪」这个唯一目的变得不可靠。

### 3.4 级别归属

| 级别 | 内容 |
| --- | --- |
| `INFO` | 原有全部 `print`：扫描开始、seed、每段 match 摘要、每块一行摘要、写出汇总 |
| `DEBUG` | 每块完整诊断字典、重试的窗口与半径、momentum 复测结果、锚点逐点 `(time, offset, score)`、每条 FFmpeg 命令行、fingerprint 索引规模 |
| `WARNING` | 低置信透传、段内未匹配、coverage deficit、FFmpeg stderr 尾巴、参考被截断 |
| `ERROR` | 块级异常后回落透传 |

DEBUG 的 FFmpeg 命令行是顺带补的能力：此前解码失败才在异常里看得到 stderr，重现问题得靠猜参数。

## 4. result

### 4.1 格式：单 JSON 文档，不是 JSONL

JSONL 的唯一优势是中断可见，而这条职责已由日志承担且做得更好（时间戳、级别、可 tail）。换来的是 header / segments / chunks / summary 能互相引用，一次 `json.load` 即可配合音频使用。

中断不丢：写出放在 `try/finally`，`KeyboardInterrupt` 与异常路径同样 dump 已积累的部分。

```json
"status": "complete" | "interrupted" | "failed" | "scan-only"
```

`chunks` 按已消费顺序追加，中断时就是「跑到哪」的完整记录。`scan-only` 下 `output` 为 `null`、`chunks` 为空数组、`summary` 只有段级统计。

### 4.2 累加器如何跨阶段

`scan_reference` 与 `process_audio` 是 CLI 分两次调用的，scan 阶段的信息（`scan_mode_used`、seed）无法经由 `list[AlignmentSegment]` 传出。

因此 `result.py` 的 `_RunResult` 由 **CLI 构造**，以内部关键字参数 `_result=` 传给两个阶段（下划线前缀即内部可改，符合包约定），CLI 的 `finally` 负责 dump。库使用者不传就不产生 result 文件，其诊断手段是 logger 与返回值。

`run.argv` 是 CLI 概念，库路径下为 `null`。

### 4.3 Schema

```jsonc
{
  "schema": "audio-overlap-removal/result@1",
  "status": "complete",

  "run": {
    "version": "0.1.0",
    "argv": ["audio-overlap-removal", "mix.webm", "..."],   // 库调用时为 null
    "started_at": "2026-09-09T04:12:00.123Z",
    "finished_at": "2026-09-09T04:25:32.881Z",
    "elapsed_sec": 812.758,
    "realtime_factor": 0.113,
    "workers": 4,
    "settings": {
      "sample_rate": 48000, "chunk_sec": 30.0, "context_sec": 1.0,
      "search_sec": 0.25, "strength": 1.0,
      "profile": {
        "cleanup_strength": 1.0, "center_strength": 1.0,
        "center_cleanup_strength": 1.0, "silence_cleanup_strength": 1.0
      },
      "profile_overridden": false,
      "adaptive_time_warp": true, "momentum": true,
      "scan_mode": "auto", "scan_mode_used": "fingerprint",
      "segments_source": "scan" | "file:seg.json"
    }
  },

  "inputs": {
    "mixture":   { "media_id": "mixture", "path": "...", "duration_sec": 7204.3,
                   "sample_rate": 48000, "channels": 2, "size_bytes": 189234112 },
    "references": [
      { "media_id": "reference", "path": "...", "duration_sec": 3600.1,
        "sample_rate": 44100, "channels": 2, "size_bytes": 91234112 }
    ]
  },

  "output": {
    "path": "out.flac", "format": "FLAC", "subtype": "PCM_24",
    "sample_rate": 48000, "channels": 2,
    "written_start_sec": 0.0, "written_end_sec": 7204.3, "duration_sec": 7204.3
  },

  "segments": [
    {
      "index": 0,
      "media_id": "reference",
      "mixture_start": 118.0, "mixture_end": 3702.5,
      "reference_start": 0.417, "reference_end": 3584.9,
      "offset_sec": 117.583, "offset_slope": -1.3e-5,
      "speed": 1.000013,
      "median_score": 0.812,
      "trajectory_deviation_sec": 0.031,
      "anchors": { /* 见 4.5 */ }
    }
  ],

  "chunks": [
    {
      "index": 3,
      "start_sec": 90.0, "end_sec": 120.0,
      "mode": "cancelled",                 // cancelled | low-confidence | unmatched
      "segment": 0,                        // 未匹配为 null
      "media_id": "reference",             // 未匹配为 null
      "reference_start_sec": 1234.567,     // core 起点的参考绝对时间
      "offset_used_sec": 117.583,
      "offset_source": "momentum" | "trajectory",
      "momentum_confident": true,

      "alignment_score": 0.874,
      "gain_p05": 0.61, "gain_median": 0.68, "gain_p95": 0.74,
      "side_corr_median": 0.55, "foreground_guard": 0.31,
      "side_residual_ratio": 0.42, "cleanup_output_ratio": 0.63,
      "control_reduction_db": -8.4,

      // 仅在触发时出现，字段名沿用现有诊断键
      "retry_radius_sec": 1.0,
      "short_warp_considered": 1.0, "short_warp_accepted": 1.0,
      "long_validation_score": 0.81, "short_validation_score": 0.86,
      "coarse_prior_error_sec": 0.004,
      "long_anchor_drift_samples": 12.0, "long_anchor_jitter_samples": 3.0,
      "coverage_deficit_start_sec": 0.0, "coverage_deficit_end_sec": 0.0
    }
  ],

  "summary": {
    "chunks": { "total": 240, "cancelled": 231, "low_confidence": 6, "unmatched": 3 },
    "seconds": { "total": 7204.3, "cancelled": 6930.0, "low_confidence": 180.0,
                 "unmatched": 94.3, "matched_coverage": 0.985 },
    "control_reduction_db": { "median": -8.1, "p05": -14.2, "p95": -2.3 },
    "alignment_score": { "median": 0.86, "min": 0.44 },
    "passthrough_spans": [[3120.0, 3180.0], [5400.0, 5430.0]],
    "warnings": ["reference shorter than segment 0 after 3584.9s"]
  }
}
```

### 4.4 字段来源

- **`media_id` 预留多参考**：`inputs.references` 是数组，segment / chunk 各带 `media_id`。当前恒为 `"reference"`，但结构不必为多参考重来一遍——`fingerprint.py` 本就有 `media_id` 概念，接得上。
- **`reference_start_sec`** = `window_start + aligned_start_samples / sr + (chunk.position - chunk.context_start)`。`cancel_attempt` 现在把 `window_start` 留在闭包里，需随诊断带出；`aligned_start` 已在诊断里，但语义是「搜索窗内样本索引」，**改名 `aligned_start_samples`** 并补注释，避免继续被误读成时间。
- **段级 `reference_start` / `reference_end`** = 两端各自减去该时刻的 `offset_at()`，因为 offset 随时间漂移，不能共用一个值。
- **`offset_source`**：区分这块用的是 momentum 复测值还是段轨迹插值——「某段整体偏了」和「某块单独偏了」是两类问题。
- **`run.version`** 读包内 `__version__` 常量，`pyproject.toml` 改为 dynamic 从此处取。`importlib.metadata` 在 PyInstaller onedir 下不可靠，而冻结分发是本项目的既有场景（见 [`packaging.md`](packaging.md)）。

### 4.5 锚点三档

摘要**不抽样**，只保留分布与极端值：诊断锚点时关心的从来不是那些好锚点，而是尾部。

`summary`（默认）：

```jsonc
"anchors": {
  "count": 240,
  "interval_sec": { "median": 15.0 },
  "score":        { "min": 0.44, "p05": 0.52, "median": 0.81, "p95": 0.93 },
  "residual_sec": { "rms": 0.004, "p95": 0.021 },
  "sparsest_gap": { "start": 2100.0, "end": 2162.3 },
  "weakest": [
    { "time": 2130.0, "offset": 117.58, "score": 0.44 }
    // 分数最低的 min(10, count) 个；同分按 time 升序，保证可复现
  ]
}
```

- **`residual_sec` 相对线性模型**（`offset_sec + offset_slope * (t - center)`），**不是**相对 `offset_at()`。后者在有锚点时是锚点插值，锚点处残差恒为 0，度量不出任何东西。段级 `trajectory_deviation_sec` 是同一度量的 max，因此这里只给 `rms` 与 `p95`，不重复 max。
- `residual` 大 = 真实漂移不是直线，线性模型跟不上；这正是锚点轨迹存在的理由，也是块级对齐失败的前兆。
- `sparsest_gap` 是最大空洞（其长度即锚点间隔的 max，故 `interval_sec` 只留 median）。洞里的块只能靠插值，最容易掉出搜索半径；与 `residual_sec` 对照着看。

`full`：在上述之上再加三条完整数组 `times` / `offsets` / `scores`。
`none`：只留 `count`。

**体积**：每锚点 3 个 float ≈ 60 B，250 ms 间隔的长段可达数万锚点，10 小时素材 `full` 最坏约 8–10 MB。`summary` 与段数同阶，可忽略。

## 5. 代码改动

| 文件 | 改动 |
| --- | --- |
| `logging_setup.py`（新，内部） | `_configure_logging(log_path, level, quiet)`；仅 CLI 调用 |
| `result.py`（新，内部） | `_RunResult` 累加器：`record_inputs/segments/chunk/finish`，`dump(path, status)`；原子写出；默认路径推导 |
| `__init__.py` | 挂 `NullHandler`；定义 `__version__` |
| `alignment.py` | 6 处 `print` → `logger.info/debug`；`anchors` 的 score 透传进 segment；scan_mode_used 记入 `_result` |
| `pipeline.py` | 8 处 `print` → logger；`cancel_attempt` 返回 `window_start`；`process_chunk` 带出 `segment` / `offset_source`；`report_path` → `_result` |
| `cancellation.py` | `aligned_start` → `aligned_start_samples` |
| `models.py` | `AlignmentSegment` 加 `anchor_scores`；序列化同步 |
| `cli.py` | 选项重整；构造 `_RunResult` 并在 `finally` dump；2 处 `print` → logger |
| `media.py` | `_probe_audio` 结果供 result；FFmpeg 命令行 `logger.debug` |
| `pyproject.toml` | version 改 dynamic |
| `README*.md` / `CLAUDE.md` | 同步 |

## 6. 破坏性变更（按项目约定不留兼容路径）

1. `--report` 删除，由 `--result` 取代；格式从 JSONL 变为单 JSON 文档。
2. 进度输出从 stdout 挪到 stderr。
3. 诊断键 `aligned_start` → `aligned_start_samples`。
4. `AlignmentSegment` 新增 `anchor_scores`。**例外**：`_segment_from_dict` 允许该键缺失（视为空），因为 seg.json 是手工编辑的输入格式，强制补一条几百项的数组只会逼人绕开这个工作流。缺失时 result 里该段 `anchors.score` 为 `null`。
5. 默认产生三个附属文件（此前一个都不产生）。

## 7. 测试

- 默认路径推导：给定 `out.flac` 得到三个 `out-*`；`--scan-only` 回落到 mixture stem；`--no-*` 三个 flag 各自生效。
- `--segments` 读入时默认不写 segments-out，显式给路径则写。
- 日志：`--log-level debug` 文件含 DEBUG 行；`--quiet` 下 stderr 仅 warning+ 而文件不受影响。
- 库直接调用 `process_audio` 不产生任何 handler 输出，也不产生 result 文件。
- result：schema 键齐全；`chunks` 数与块数一致；`segment` / `media_id` 在 unmatched 块为 `null`；`--scan-only` 下 `output` 为 `null`。
- `reference_start_sec` 正确性：合成已知 offset 的素材，断言逐块还原出的参考时间与真值差 < 一个块内搜索半径。
- `residual_sec` 非零：构造非线性漂移的锚点轨迹，断言 rms > 0（防止退回相对 `offset_at()` 计算）。
- 中断：处理途中抛异常，断言 result 落盘且 `status == "failed"`、`chunks` 非空。
- `--result-anchors none/summary/full` 三档字段差异；`weakest` 确实是分数最低的那些。

## 8. 被否决的替代方案

- **result 继续用 JSONL、每行带 `type` 字段**：能保留流式，但要读的一方自行按 type 重组，「配合音频直接用」的目标反而更远；中断可见性已由日志覆盖。
- **`--log` 用 `nargs="?"` 无值触发默认名**：与位置参数相邻时歧义（`--log mix.webm` 会把 mixture 吃掉）。
- **把 result 并进 `--segments-out`**：一个是可编辑输入、一个是只读产物，合并会让「改完 offset 再跑」的工作流每次都得手工剔除大量结果字段。
- **`logging` 的 `QueueHandler` + listener**：并发是 `ThreadPoolExecutor`（见 `parallel.py`），无必要，纯粹多一层。
