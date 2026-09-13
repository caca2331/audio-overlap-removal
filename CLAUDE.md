# CLAUDE.md

## 项目

从真实混音 `C` 中消除已知参考媒体 `B`（`C = A + B'`，`B'` 经过增益、
EQ、有损编码、暂停/跳转/重放和轻微变速）。不是通用人声分离器：无法可靠
匹配的区间一律原样透传。

## 命令

```bash
python -m pytest test_audio_overlap_removal.py -q
```

```bash
ruff check audio_overlap_removal test_audio_overlap_removal.py
```

需要 `ffmpeg` 与 `ffprobe` 在 `PATH` 上；涉及真实解码的测试会在缺失时
自动跳过。

## 结构与数据流

`cli.py` → `pipeline.py` 编排；`alignment.py` 定位与块内对齐；
`cancellation.py` 纯内存 DSP；`fingerprint.py` 长媒体索引；
`media.py` 所有 FFmpeg 解码与原子写出；`models.py` 公共数据模型；
`parallel.py` 有界保序并行。

扫描（`scan_reference`）与处理（`process_audio`）可分开调用。处理阶段
是两遍：先低采样率逐块复测 offset 并拟合轨迹，再按该轨迹居中解码原生
采样率参考窗做消除。

## 长任务工作流（开发用，未写入 README）

真实素材一跑就是几十分钟，这几个选项用于把它拆开、观察和续跑：

```bash
audio-overlap-removal mix.webm ref.webm --scan-only
```

```bash
audio-overlap-removal mix.webm ref.webm out.flac --segments mix-segments.json --log-level debug
```

- `--scan-only` / `--segments-out` / `--segments`：扫描与消除解耦。JSON 含
  完整锚点轨迹，可手工修改 offset 后再跑，不必每次重扫。
- 日志与 result 默认写在输出音频旁边（`out-log.txt` / `out-result.json` /
  `out-segments.json`），跑到哪一块看日志、跑完分析看 result。细节见
  [`docs/diagnostics.md`](docs/diagnostics.md)。
- `--output-start/--output-end`：只写指定区间（对齐 context 仍可读取区间
  外）。断点续传即用它分段跑，再 `ffmpeg -f concat` 拼接；当前没有自动
  checkpoint——原子写出是全有或全无的。分段跑时输出音频名不同，附属文件随之
  分开，不会互相覆盖。

## 约定

- 包根导出的名字是公共接口；下划线前缀的一律内部可改。
- 单个块的失败绝不能终止整条长音频：返回状态、扩窗重试、最后才透传。
- 块内搜索必须带先验约束——在整窗口做无约束 argmax 会把重复段落变成
  高置信度的错误答案。
- 判断一个块是否该采用消除结果，用多个相互独立的证据表决，不要依赖
  单一相关分数。
- 透传路径必须是逐样本原样复制，不做任何混合或斜坡。
- 注释解释「为什么」，不复述代码；不保留向后兼容的旧路径。

## 任务相关文档

- [`docs/real-world-goals.md`](docs/real-world-goals.md)：真实素材、生产
  优先级、`--strength` 刻度语义与验收准则。调参或改默认值前先读。
- [`docs/algorithm-review.md`](docs/algorithm-review.md)：算法审查、能力
  边界、已被否决的替代方案（MDF/PBFDAF、Wiener 掩码、2×2 Side-MIMO 等）
  及其实测数据。提出「换个相消器」之前先读，避免重复已做过的实验。
- [`README.md`](README.md) / [`README.zh-CN.md`](README.zh-CN.md)：面向
  用户的完整选项、格式与声道矩阵。改 CLI 或行为时同步。
- [`docs/packaging.md`](docs/packaging.md)：PyInstaller onedir 独立分发的
  构建方式、三平台注意事项与 FFmpeg 查找顺序。
- [`docs/diagnostics.md`](docs/diagnostics.md)：日志与 result JSON 的职责
  切分、默认文件名、完整 schema 与锚点三档。改诊断输出前先读。
- [`experiments/artifact-bench/`](experiments/artifact-bench/)：有真值的消除
  基准（真实参考窗 + 合成播出链 + 真实主播前景）与真实素材 A/B 脚本。改
  对齐或相消逻辑前后各跑一次，数字记入 `docs/algorithm-review.md`。
