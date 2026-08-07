# 独立可执行分发

面向「下载解压即用、不装 Python」的用户。FFmpeg **不打进包里**——它是
系统级依赖，用户仍需自行安装，与源码运行的前置条件保持一致。

## 形态

PyInstaller **onedir**（一个目录，打包成压缩档分发），不是 onefile。

onefile 每次启动都要把全部内容解压到临时目录，且无法避免：numpy、scipy、
libsndfile 都是原生扩展，操作系统的动态链接器（`LoadLibrary` / `dlopen`）
只接受真实文件路径，没有从内存加载的可移植途径。onedir 把这次解压变成
用户手动的一次性动作，之后每次启动都是零开销，出问题时目录内容也可见可查。

## 构建

PyInstaller 不能交叉编译，三个平台各构建一次。

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

- Windows x64、macOS：本机按上面的流程。macOS 只能得到构建机自身的架构，
  numpy/scipy 没有 universal2 wheel，`--target-arch universal2` 走不通，
  Intel 与 Apple Silicon 需要两台机器（或两个 runner）；
- Linux x64：走容器，见下。

### Linux

在 [`Dockerfile.manylinux`](../packaging/Dockerfile.manylinux) 定义的镜像里
构建，图的是它 **glibc 2.28** 的底座——在老 glibc 上链接的二进制能在更新的
发行版上跑，反过来不行。实测符号版本上限正好落在 2.28，覆盖 RHEL 8、
Debian 10+、Ubuntu 18.10+。

```bash
docker build -t audio-overlap-removal-build:manylinux -f packaging/Dockerfile.manylinux packaging
```

```bash
docker run --rm -v "$PWD:/src" -w /src audio-overlap-removal-build:manylinux bash packaging/build-manylinux.sh
```

镜像里额外 `dnf install python3.11`，因为 manylinux 自带的 `/opt/python/*`
是静态编译的，没有 `libpython.so`，PyInstaller 会直接拒绝。

**Linux 产物必须拿到构建容器之外的发行版上、用那台机器自己的 FFmpeg 验证**
（见下一节）。容器内 FFmpeg 与 bundle 同源，测不出真实环境的问题。

## 体积

实测：Windows x64 解包 **128.8 MB** / 压缩 **52.8 MB**，Linux x64 解包
**208.6 MB** / 压缩 **57.7 MB**（Linux 多出 libgfortran、libstdc++ 等运行时
库）。Windows 侧构成：

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
- `ssl` 已排除（约 1 MB）。`_hashlib` 看着也是死重量（还能再省 5 MB，
  应用确实从不用它），但 Linux 上 PyInstaller 会加 `pkg_resources` 运行时
  钩子，它在 `main()` 之前 import `_hashlib`，排除后程序启动即崩——
  Windows 上不引入这个钩子所以测不出来。**往列表里加东西前，每个平台都要
  实测**，4% 的体积不值得换平台相关的脆弱性。
- **不要开 UPX**：能再砍掉约一半解包体积，但会破坏部分 numpy/scipy 的
  DLL，并显著抬高杀软误报率。

再往下只有去掉 scipy 依赖（自行实现 FFT 卷积、中值滤波、KD 树查询）能把
总量压到 60 MB 量级，那是算法改造，不属于打包范畴。

## 用户侧的坑

- **macOS**：未签名二进制被 Gatekeeper 拦截。至少 ad-hoc 签名
  `codesign -s - --force --deep dist/audio-overlap-removal`，并在发布说明里
  给出 `xattr -dr com.apple.quarantine <目录>`。要彻底干净需要 Apple
  开发者账号做公证。
- **Windows**：SmartScreen 会对未签名程序告警；PyInstaller 产物也是杀软
  误报的常见对象。

## 子进程的 loader 路径

Linux/macOS 上 PyInstaller 会给冻结进程注入
`LD_LIBRARY_PATH=<bundle>/_internal`（macOS 对应 `DYLD_LIBRARY_PATH`）。
FFmpeg 子进程继承它之后，会去加载 bundle 里那份构建容器编译的
`libstdc++.so.6`，缺 `GLIBCXX_3.4.29` 而无法启动——系统 FFmpeg 越新越容易
撞上。`media.py` 的 `_child_env()` 在 spawn 前把这个变量还原成
PyInstaller 存下的 `*_ORIG`（原本没有就直接删掉）。

这个缺陷只在 Linux 打包版 + 系统 FFmpeg 的组合下出现，构建容器内部因为
两者同源而测不出来。

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
