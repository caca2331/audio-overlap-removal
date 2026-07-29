# Audio Overlap Removal

从混合音频中消除一条已知、内容同源但时间轴可能不同的参考音轨。

典型场景是直播或录屏：

- **C（mixture）**：主播声音 + 游戏/BGM；
- **B（reference）**：干净的游戏/BGM 原轨；
- **A'（output）**：尽量保留主播声音、消除 B 的结果。

可近似写成 `C = A + B'`。其中 B' 与 B 内容相同，但可能经历音量变化、
编码损失、EQ、暂停、跳转、重放或轻微播放速度漂移。本项目先寻找 B 在 C
中的分段位置，再进行参考信号相消和受保护的残留清理。

> 这不是通用的人声/伴奏分离器。若参考文件与混合音中的背景内容不同，
> 算法会原样通过低置信度区间，而不能“猜出”应当移除的声音。

## 功能概览

- 自动扫描参考音轨在混合音频中的位置；
- 暂停、续播、跳转和重放后可重新获取；
- 250 ms 局部时间扭曲，并可在验证通过后采用 16 ms 精细路径；
- 分块解码和处理，避免按原始采样率一次性载入完整媒体；
- 支持 mono、stereo，以及由 FFmpeg 下混的多声道输入；
- 保留混合输入的 mono/stereo 布局；
- 可并行扫描和处理，输出顺序与单线程一致；
- 输入解码交给 FFmpeg，输出支持 24-bit FLAC 和 WAV。

当前推荐入口是 `real_reference_cancel.py`，安装后也可直接使用
`audio-overlap-removal` 命令。

## 安装

要求：

- Python 3.10 或更高版本；
- `ffmpeg` 和 `ffprobe` 可在 `PATH` 中运行；
- 推荐至少 8 GiB 内存；提高 `--workers` 会增加峰值内存。

创建虚拟环境并安装：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -e .
```

确认外部解码器可用：

```bash
ffmpeg -version
ffprobe -version
```

## 快速开始

自动扫描并处理完整混合音频：

```bash
audio-overlap-removal mixture.webm reference.webm clean.flac
```

不安装命令行入口也可以直接运行：

```bash
python real_reference_cancel.py mixture.webm reference.webm clean.flac
```

只处理混合音频从第 300 秒开始的 15 分钟：

```bash
audio-overlap-removal \
  mixture.webm reference.webm clean.flac \
  --start 300 --duration 900 \
  --strength 1 --workers 4
```

已知固定偏移时，可跳过全局扫描。`--offset` 定义为
“混合时间减参考时间”；例如混合的第 81.533 秒对应参考的第 0 秒：

```bash
audio-overlap-removal \
  mixture.webm reference.webm clean.wav \
  --offset 81.533 \
  --start 600 --duration 120
```

输出路径必须以 `.flac` 或 `.wav` 结尾，也不能与任一输入文件相同。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--offset SECONDS` | 自动扫描 | `C 时间 - B 时间`；指定后跳过全局扫描 |
| `--start SECONDS` | `0` | 从混合输入的哪个时间开始处理 |
| `--duration SECONDS` | 到文件末尾 | 输出时长 |
| `--chunk SECONDS` | `30` | 处理块长度 |
| `--workers N` | `1` | 并行扫描/处理任务数 |
| `--sample-rate HZ` | `48000` | 解码、处理和输出采样率 |
| `--strength VALUE` | `0` | 保真与消除强度的统一控制 |
| `--disable-adaptive-warp` | 关闭 | 禁用经验证的 16 ms 精细时间扭曲 |

`--strength` 没有硬上限：

| 值 | 取向 |
| ---: | --- |
| `0` | 最保守，只做受保护的参考相消 |
| `0–1` | 逐步增加残留和居中媒体清理 |
| `1` | 较激进，适合背景仍明显的素材 |
| `1.25–2` | 更偏向媒体去除或 ASR，低语损失和失真风险更高 |
| `>2` | 实验范围；继续增强清理，也继续增加损伤风险 |

`--cleanup-strength`、`--center-strength`、
`--center-cleanup-strength` 和 `--silence-cleanup-strength` 是专家级覆盖项。
通常先只调整 `--strength`。

## 格式与声道兼容性

### 输入

程序通过 FFmpeg 解码第一个音频流，因此通常可读取：

- WAV、FLAC、AIFF；
- MP3、AAC/M4A；
- Ogg Vorbis、Opus；
- WebM、常见带音频的视频容器；
- 当前 FFmpeg 构建支持的其他非 DRM 格式。

采样率和采样格式会统一转换，不要求两路输入一致。损坏文件、加密/DRM
媒体、FFmpeg 构建未包含的编解码器，以及没有音频流的文件不受支持。
解码失败时，程序会保留 FFmpeg 的错误详情。

### 声道

| 原始输入 | 内部处理与输出 |
| --- | --- |
| mixture 为 mono | mono 处理，输出 mono |
| mixture 为 stereo | Mid/Side 处理，输出 stereo |
| mixture 超过 2 声道 | FFmpeg 下混为 stereo，再输出 stereo |
| reference 为 mono | 使用 mono/Mid 参考路径 |
| reference 为 stereo | 使用 Mid/Side 参考路径 |
| reference 超过 2 声道 | FFmpeg 下混为 stereo 后作为参考 |

多声道下混会丢失原始环绕布局。如果必须保留 5.1/7.1，请先自行拆分和路由
声道；当前算法只建模 mono/stereo。

### 输出

- `.flac`：FLAC 容器，24-bit PCM；
- `.wav`：WAV 容器，24-bit PCM。

当前不会复制输入的封面、视频、章节或其他元数据，只输出处理后的音频。

## 工作原理

1. **低采样率全局扫描**：在 1 kHz 音频上寻找可靠种子和候选偏移；
2. **分段跟踪与重获取**：局部跟踪失败后限频执行全局搜索，处理暂停、跳转
   和重放；
3. **局部时间对齐**：用锚点构造时间扭曲，必要时验证更密集的候选路径；
4. **Mid/Side 参考相消**：利用较少受居中主播人声影响的 Side 估计复数传递
   函数，并结合参考 Mid 处理居中媒体；
5. **受保护的残留清理**：检测主播声音或与参考无关的立体声内容，自动减弱
   后级抑制；
6. **分块写出**：按时间顺序写出并平滑块边界。

更深入的能力边界和实验结论见：

- [`docs/algorithm-review.md`](docs/algorithm-review.md)
- [`docs/real-world-goals.md`](docs/real-world-goals.md)

## 性能建议

- 先用 `--duration 30` 或 `--duration 120` 做短片段试听；
- 一般从 `--workers 1` 或 `2` 开始；
- 内存充足时可尝试 `--workers 4`；
- worker 数过高通常受内存带宽限制，并会近似按并发块数增加临时内存；
- 已知偏移时使用 `--offset`，可省去全局扫描。

默认全局扫描仍需以低采样率解码完整参考音轨。非常长的参考媒体会增加扫描
时间和内存，但不会以 48 kHz 原始声道布局整体载入。

## 测试

运行完整回归测试：

```bash
python -m unittest -v test_real_reference_cancel.py
```

测试覆盖动态增益、暂停、跳转/重放、速度漂移、mono/stereo 路由、多声道
FFmpeg 下混、WAV/FLAC 输出选择、截断参考和并行结果顺序。

真实媒体和历史输出体积较大，不包含在此独立项目中。若要运行旧的素材生成
脚本，请把本地文件放到：

```text
extra-asset/
├── miyako.webm
├── sr.webm
└── ya.mp3
```

这些目录已被 `.gitignore` 排除。

## 已知限制

- 参考音轨必须与混合音中的目标背景同源；仅风格相似不够；
- 强压缩、重混音、复杂动态处理或大幅变速会降低匹配和相消效果；
- mono 路径没有安全的 Side 控制信号，因此比 stereo 更保守；
- 两个文件都是 stereo、但背景在混入前已下混为 mono 的情况不会单独检测；
- 与参考无关且高度居中的背景，也可能被激进清理误伤；
- 当前只选择每个输入的第一个音频流；
- 不保留多声道环绕布局或媒体元数据。

低置信度区间会直接通过，优先避免错误相消。如果输出背景残留较多，可先确认
参考内容和偏移正确，再逐步提高 `--strength`。

## 项目结构

```text
audio-overlap-removal/
├── real_reference_cancel.py       # 当前推荐实现和 CLI
├── test_real_reference_cancel.py  # 算法与 I/O 回归测试
├── pyproject.toml                 # 依赖与命令行入口
├── docs/                          # 目标、实验结论和算法审查
├── main.py                        # 旧 mono 频谱实验
├── evaluate.py                    # 旧合成评估
├── generate_tests.py              # 旧素材生成脚本
├── run_tests.py                   # 旧生成/评估入口
└── scratch/                       # 历史分析脚本
```

旧实验脚本依赖 `librosa`，分析脚本还需要 `matplotlib`。仅在需要它们时安装：

```bash
python -m pip install -e ".[legacy]"
```
