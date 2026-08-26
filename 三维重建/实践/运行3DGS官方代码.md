克隆官方代码：

```cmd
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

所使用的 Python 环境中除了 PyTorch 等相关依赖之外，还需要：

```cmd
pip install plyfile tqdm opencv-python
```

进入克隆下来的代码所在的目录，编译底层 CUDA 拓展模块：

```cmd
cd gaussian-splatting #进入克隆的代码所在目录
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

下载官方场景数据：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

运行代码，例如：

```cmd
python train.py -s data/tandt/train
```

训练完后，可以使用官方查看器查看，官方查看器预编译下载：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip