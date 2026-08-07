# 独立可执行分发

面向「下载解压即用、不装 Python」的用户，目标平台是 **Windows 与 macOS**。
FFmpeg **不打进包里**——它是系统级依赖，用户仍需自行安装，与源码运行的
前置条件保持一致。

不做 Linux 分发：用 Linux 的人自己从 PyPI 装就是了，而为它出独立包要额外
维护一个老 glibc 的构建容器，外加在第二个发行版上跑验证。macOS 只发
Apple Silicon，理由见下。

## 形态

PyInstaller **onedir**（一个目录，打包成压缩档分发），不是 onefile。

onefile 每次启动都要把全部内容解压到临时目录，且无法避免：numpy、scipy、
libsndfile 都是原生扩展，操作系统的动态链接器（`LoadLibrary` / `dlopen`）
只接受真实文件路径，没有从内存加载的可移植途径。onedir 把这次解压变成
用户手动的一次性动作，之后每次启动都是零开销，出问题时目录内容也可见可查。

## 构建

PyInstaller 不能交叉编译，每个目标各构建一次。日常发布交给
[`.github/workflows/build.yml`](../.github/workflows/build.yml)，下面这套
是本机复现用的——CI 红了要排查时，本机的迭代循环比每次等十分钟快得多。

**必须在干净的虚拟环境里构建。** PyInstaller 分析的是解释器实际能看到的
site-packages，共享环境（尤其是装了 ML 栈的 conda base）会被静态分析连坐
拖进来：本项目在一个这样的环境里试构建，产出 7.4 GB / 7400 个文件，包含
torch、paddle、onnxruntime、transformers——运行期一个都不会 import。
`build.py` 因此设了 400 MB 的体积上限，超过就直接失败并提示这件事。

```bash
python -m venv .venv-build
```

```bash
.venv-build/bin/python -m pip install -e ".[build]"
```

```bash
.venv-build/bin/python packaging/build.py
```

Windows 上把 `.venv-build/bin/python` 换成 `.venv-build\Scripts\python.exe`。
`build.py` 用 `sys.executable -m PyInstaller` 调用，所以不需要先 activate。

验收标准：冻结版与源码版在同一份输入上应当**逐字节一致**。这是最省事的
回归证明——数值有任何偏差都说明打包动到了不该动的东西。

产出在 `dist/`：可执行目录、按平台命名的压缩档（Windows 用 zip，其余用
tar.gz，因为 zip 条目不保留可执行位）、以及 `.sha256`。构建脚本会跑一次
`--help` 冒烟测试，走完整个 import 链——`excludes` 砍错东西会在这里暴露。

macOS 只发 Apple Silicon。构建机只能产出自身架构，numpy/scipy 没有
universal2 wheel，`--target-arch universal2` 走不通，Intel 版需要单独一台
机器；而 GitHub 的 Intel runner 稀缺到能把整个 run 拖上近一小时。Intel Mac
用户从 PyPI 装即可。

## 体积

CI 实测：

| 目标 | 解包 | 压缩档 |
| --- | --- | --- |
| windows-x86_64 | 129.1 MB | 53.0 MB |
| macos-arm64 | 91.2 MB | 29.6 MB |

macOS 小 38 MB，因为 Apple Silicon 上 numpy/scipy 用系统的 Accelerate 做
BLAS，不必各自捆一份 OpenBLAS。Windows 侧构成：

| 组成 | MB |
| --- | --- |
| scipy | 47.5 |
| numpy.libs（OpenBLAS） | 20.0 |
| scipy.libs（OpenBLAS） | 19.3 |
| numpy | 6.0 |
| python3xx.dll | 6.6 |
| libsndfile | 2.3 |
| 其余（stdlib 扩展、运行时） | ~22 |

96% 是原生二进制，这基本就是地板。几条被验证过的结论：

- **不要往 `EXCLUDES` 里加 scipy 子包。** 实测 `import scipy.signal` 会
  立刻拉进 `constants/fft/integrate/interpolate/linalg/ndimage/optimize/
  signal/sparse/spatial/special/stats` 全部子包，不是懒加载——那 66.8 MB
  一个字节都省不掉，砍掉必然在运行期崩。
- numpy 和 scipy 各带一份 OpenBLAS（39 MB），编译产物与符号后缀不同，
  不能去重。
- `ssl` 已排除（约 1 MB）。`_hashlib` 还能再省 5 MB，应用也确实从不用它，
  但排除它曾让 Linux 构建启动即崩：PyInstaller 在那边会加 `pkg_resources`
  运行时钩子，在 `main()` 之前 import 它，而 Windows 不引入这个钩子，所以
  当时是在 Linux 的冒烟测试里才暴露。Linux 已不是发布目标，这 5 MB 因此
  可以重新评估，但**要在 Windows 和 macOS 上都实测通过**才能动。
- **不要开 UPX**：能再砍掉约一半解包体积，但会破坏部分 numpy/scipy 的
  DLL，并显著抬高杀软误报率。

再往下只有去掉 scipy 依赖（自行实现 FFT 卷积、中值滤波、KD 树查询）能把
总量压到 60 MB 量级，那是算法改造，不属于打包范畴。

## 用户侧的坑

- **macOS 会说程序「已损坏」，这是必然发生的，且与打包无关。** PyInstaller
  的产物带 ad-hoc 签名但没有公证票据，凡是被 quarantine 标记的这类程序，
  Gatekeeper 一律报 damaged——实测换归档格式、换解压方式都不影响，清掉
  `com.apple.quarantine` 就能启动。README 里已把这一步写在显著位置。想彻底
  免掉它需要 Apple 开发者账号（Developer ID 签名 + 公证），没有别的办法。
- **macOS 的产物用 `ditto` 打 zip，不要换回 tar.gz。** 归档本身没问题，
  但 The Unarchiver 解不开它——会在 numpy dist-info 下面的空目录条目上报
  「无法打开文件」。用户装了什么解压工具不该决定下载能不能用。
- **Windows**：SmartScreen 会对未签名程序告警；PyInstaller 产物也是杀软
  误报的常见对象。

## 子进程的 loader 路径

类 Unix 平台上 PyInstaller 会给冻结进程注入指向 bundle 自身的
`DYLD_LIBRARY_PATH`（Linux 上是 `LD_LIBRARY_PATH`）。FFmpeg 子进程继承它
之后，会拿 bundle 里那份构建时的库去解析自己的依赖，系统 FFmpeg 越新越
容易撞上版本不匹配而无法启动。`media.py` 的 `_child_env()` 在 spawn 前把
这个变量还原成 PyInstaller 存下的 `*_ORIG`（原本没有就直接删掉）。

这个缺陷是在 Linux 上发现的（系统 ffprobe 拿 bundle 的 `libstdc++.so.6`
去解析，缺 `GLIBCXX_3.4.29`）。macOS 的机制相同，所以修复对两边都保留。
它只在打包版 + 系统 FFmpeg 的组合下出现，构建机上因为两者同源而测不出来
——**首次发布 macOS 版前，务必在一台装了 Homebrew FFmpeg 的 Mac 上实跑**。

## FFmpeg 查找顺序

见 [`media.py`](../audio_overlap_removal/media.py) 的 `_tool()`：

1. `AOR_FFMPEG_DIR` 指定的目录；
2. 冻结运行时，可执行文件所在目录及其 `bin/` 子目录；
3. `PATH`；
4. 各平台常见安装位置（Windows：`Program Files\ffmpeg\bin`、`C:\ffmpeg\bin`、
   Chocolatey、Scoop、WinGet Links；macOS：`/opt/homebrew/bin`、
   `/usr/local/bin`、MacPorts；Linux：`/usr/local/bin`、`/snap/bin`、
   linuxbrew、`~/.local/bin`）。

第 3 步排在第 4 步之前是刻意的：用户自己配好的 `PATH` 不该被猜测出来的
路径覆盖。第 2 步让「把 ffmpeg 丢进解压目录」成为一种受支持的安装方式，
也意味着将来若要提供自带 FFmpeg 的整合包，只是分发期打包多放两个文件，
不需要改代码。
