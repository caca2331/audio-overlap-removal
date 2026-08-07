# Audio Overlap Removal

[English](README.md) | **简体中文**

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
- 分段 offset 保留实测轨迹而非单一斜率，消除前再用低采样率逐块复测；
- 单块对齐失败时自动扩大参考窗重试，而不是中断整个任务；
- 250 ms 局部时间扭曲，并可在验证通过后采用 16 ms 精细路径；
- 分块解码和处理，避免按原始采样率一次性载入完整媒体；
- 支持 mono、stereo，以及由 FFmpeg 下混的多声道输入；
- 保留混合输入的 mono/stereo 布局；
- 可并行扫描和处理，输出顺序与单线程一致；
- 输入解码交给 FFmpeg，输出支持 24-bit FLAC 和 WAV。

安装后推荐使用 `audio-overlap-removal` 命令；Python 项目也可以直接导入
`audio_overlap_removal` 包。

## 安装

### 独立可执行版

Windows x64 和 Apple Silicon macOS 提供免装 Python 的构建。从
[最新 release](https://github.com/caca2331/audio-overlap-removal/releases/latest)
下载对应平台的压缩包，解压后运行其中的 `audio-overlap-removal` 即可。仍然
需要 FFmpeg，见 [FFmpeg 的查找位置](#ffmpeg-的查找位置)。

Linux 和 Intel macOS 不提供独立版，请按下面的方式从源码安装。

**macOS 首次启动会提示程序「已损坏」。它没有损坏。** 程序是有签名的，但
Apple 公证需要付费开发者账号；凡是从网络下载、又没有公证票据的程序，macOS
一律这么报。对解压出来的目录清一次隔离属性即可：

```bash
xattr -dr com.apple.quarantine audio-overlap-removal-<版本>-macos-arm64
```

macOS 的压缩包请用访达或 `ditto -x -k` 解压，部分第三方解压工具读不对。

### 从源码安装

要求：

- Python 3.10 或更高版本；
- `ffmpeg` 和 `ffprobe` 可在 `PATH` 中运行；
- mixture 和 reference 各自不超过 24 小时；
- 内存：默认 `--workers 4` 时，一般任务 3 GB 以内，24 小时音频 8 GB 以内，
  详见[性能建议](#性能建议)。

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

### FFmpeg 的查找位置

通常放进 `PATH` 即可，但查找不止于此。`ffmpeg` 和 `ffprobe` 按以下顺序定位：

1. 环境变量 `AOR_FFMPEG_DIR` 指定的目录；
2. 可执行文件所在目录及其 `bin/` 子目录（仅独立打包版）；
3. `PATH`；
4. 各平台常见安装位置：Windows 的 `C:\Program Files\ffmpeg\bin`、
   `C:\ffmpeg\bin`、Chocolatey、Scoop、WinGet；macOS 的 `/opt/homebrew/bin`、
   `/usr/local/bin`、MacPorts；Linux 的 `/usr/local/bin`、`/snap/bin`、
   linuxbrew、`~/.local/bin`。

已配置的 `PATH` 始终优先于第 4 步猜测的位置。把 FFmpeg 复制到这些目录时要
整份复制：shared 版的编解码器在同目录的 `av*` 动态库里，只拷 `ffmpeg` 和
`ffprobe` 两个可执行文件是起不来的。

FFmpeg 装在别处时用 `AOR_FFMPEG_DIR` 指定：

```bash
AOR_FFMPEG_DIR=/opt/ffmpeg/bin audio-overlap-removal mixture.webm reference.webm clean.flac
```

## 快速开始

自动扫描并处理完整混合音频：

```bash
audio-overlap-removal mixture.webm reference.webm clean.flac --strength 1
```

在源码目录中也可以通过模块入口运行：

```bash
python -m audio_overlap_removal \
  mixture.webm reference.webm clean.flac \
  --strength 1
```

如果只有混合音频的第 300–1200 秒包含参考媒体，只扫描和处理这个范围：

```bash
audio-overlap-removal \
  mixture.webm reference.webm clean.flac \
  --start 300 --end 1200 \
  --strength 1 --workers 4
```

`--start` 和 `--end` 是 mixture 绝对时间轴上的媒体存在范围，并不是输出
裁剪范围。输出始终覆盖完整 mixture；范围外、范围内未发现参考匹配以及低
置信度区间都直接通过，只有可靠匹配到的部分会被重建。

输出路径必须以 `.flac` 或 `.wav` 结尾，也不能与任一输入文件相同。

**推荐从 `--strength 1` 开始试听。** 如果人声损伤明显，再向 `0` 调低；
如果背景残留仍多，再尝试 `1.25–2`。

## 作为 Python 库调用

大多数调用方只需要使用高层函数 `remove_reference()`：

```python
from audio_overlap_removal import remove_reference

segments = remove_reference(
    "mixture.webm",
    "reference.webm",
    "clean.flac",
    start=300,
    end=1200,
    strength=1,
    workers=4,
)
```

返回值是实际匹配到的 `AlignmentSegment` 列表。输出仍覆盖完整 mixture，
只有这些匹配区间会被处理。

如果要把扫描与处理拆开，可分别调用：

```python
from audio_overlap_removal import process_audio, scan_reference

segments = scan_reference(
    "mixture.webm",
    "reference.webm",
    start=300,
    end=1200,
    workers=4,
)
process_audio(
    "mixture.webm",
    "reference.webm",
    "clean.wav",
    alignment_segments=segments,
    strength=1,
    workers=4,
)
```

`scan_reference()` 只负责定位参考媒体，`process_audio()` 只处理传入的匹配
区间。包根目录导出的名字是稳定公共接口；以下划线开头的函数属于内部实现，
不保证跨版本兼容。

底层还导出 `fingerprint_media()`、`fingerprint_blocks()` 和
`FingerprintIndex`。索引候选包含 `media_id` 和媒体内时间，可在内存中同时
索引多条媒体；当前命令行只建立单 reference 索引，尚不持久化媒体库。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--start SECONDS` | `0` | mixture 中可能包含参考媒体的起点 |
| `--end SECONDS` | mixture 末尾 | mixture 中可能包含参考媒体的终点 |
| `--chunk SECONDS` | `30` | 处理块长度 |
| `--search SECONDS` | `0.25` | 每块的参考搜索半径，失败后自动扩大重试 |
| `--workers N` | `4` | 并行扫描/处理任务数 |
| `--sample-rate HZ` | `48000` | 解码、处理和输出采样率 |
| `--strength VALUE` | `1` | 保真与消除强度的统一控制 |
| `--disable-adaptive-warp` | 关闭 | 禁用经验证的 16 ms 精细时间扭曲 |
| `--disable-momentum` | 关闭 | 跳过消除前逐块复测 offset 的低采样率扫描 |

`--strength` 没有硬上限：

| 值 | 取向 |
| ---: | --- |
| `0` | 最保守，只做受保护的参考相消 |
| `0–1` | 逐步增加残留和居中媒体清理 |
| `1` | 推荐起点；在消除效果和目标声音保护之间取平衡 |
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

没有单独的 `--format` 参数；程序根据输出文件扩展名选择格式：

- `.flac`：FLAC 容器，24-bit PCM；
- `.wav`：WAV 容器，24-bit PCM。

当前不会复制输入的封面、视频、章节或其他元数据，只输出处理后的音频。
写出期间会在输出目录创建一个隐藏的 `.part` 临时文件；成功关闭后原子替换为
目标文件，失败时自动删除，从而避免留下半截输出。当前扫描和解码不创建其他
磁盘临时文件。

## 工作原理

1. **低采样率全局扫描**：不超过 4 小时的输入沿用完整相关搜索，以保持现有
   输出行为；更长输入流式生成紧凑指纹索引；
2. **分段跟踪与重获取**：局部跟踪失败后限频执行全局搜索，处理暂停、跳转
   和重放；每个分段保留实测的锚点轨迹，长分段不再被压缩成单一 offset 和
   斜率；
3. **Offset momentum**：正式消除之前，先在 4 kHz、较宽窗口内逐块复测
   offset，中值滤波成连续轨迹；探测不可靠的块直接继承邻块结果，而不是自
   行猜测；
4. **局部时间对齐**：块内搜索以该预测为中心展开，而非在整个窗口里取最大
   相关；随后用锚点构造时间扭曲，必要时验证更密集的候选路径；
5. **Mid/Side 参考相消**：利用较少受居中主播人声影响的 Side 估计复数传递
   函数，并结合参考 Mid 处理居中媒体；
6. **受保护的残留清理**：检测主播声音或与参考无关的立体声内容，自动减弱
   后级抑制；
7. **逐块消除后验证**：由相关分数、momentum 探测和"实际消掉多少能量"三者
   投票决定是否采用消除结果；未通过的块先扩大参考窗重试一次，仍不通过才
   透传；
8. **分块写出**：按时间顺序写出并平滑块边界，结束时汇总所有低置信度区间。

更深入的能力边界和实验结论见：

- [`docs/algorithm-review.md`](docs/algorithm-review.md)
- [`docs/real-world-goals.md`](docs/real-world-goals.md)

## 性能建议

- 已知媒体只出现在部分时间时，务必用 `--start/--end` 缩小扫描范围；
- 输出仍会重建完整 mixture，缩小扫描范围不会缩短输出；
- `--workers` 不要超过 CPU 核心数：再往上不会更快，只会多占内存。

### 内存

峰值内存（GB）大致为

```
max(0.2 + 0.8 * workers, scan)
```

其中 `scan` 在四小时以内为 `0.6 * 小时数`，超过四小时为
`0.1 + 0.025 * 小时数`——更长的输入会切换到流式索引，内存反而**更少**。
`--chunk` 和 `--sample-rate` 按比例影响 worker 那一项。

默认 `--workers 4` 时，任何时长都约为 3.4 GB。内存紧张时优先调小
`--workers`，超过 4 之后也换不来多少速度。

## 测试

运行完整回归测试：

```bash
python -m unittest -v test_audio_overlap_removal.py
```

测试覆盖动态增益、暂停、跳转/重放、速度漂移、短/长媒体扫描策略、流式
FFmpeg 解码、多媒体 ID 指纹索引、mono/stereo 路由、多声道下混、WAV/FLAC
输出选择、截断参考和并行结果顺序。

长媒体 smoke test 不生成数小时文件，而是在 96 秒夹具中压缩连续播放、动态
增益/EQ、暂停续播、向后重放和纯前景区间，并强制 FFT 与指纹两条路径做 A/B；
24 小时容量则按实际索引每窗口字节数外推并设上限断言。

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
├── audio_overlap_removal/
│   ├── alignment.py       # 参考媒体扫描与时间轴匹配
│   ├── cancellation.py    # 信号对齐、相消与残留清理
│   ├── fingerprint.py     # 流式指纹和多媒体候选索引
│   ├── media.py           # FFmpeg 解码、探测和原子写出
│   ├── models.py          # 公共数据模型与强度配置
│   ├── parallel.py        # 有界、保序的并行执行
│   ├── pipeline.py        # 可独立调用的高层处理流程
│   └── cli.py             # 命令行参数与入口
├── test_audio_overlap_removal.py  # 算法、模块接口与 I/O 回归测试
├── pyproject.toml                 # 包配置、依赖与命令行入口
└── docs/                          # 目标、实验结论和算法审查
```
