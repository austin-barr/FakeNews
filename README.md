# FakeNews
Repo for research project on improving graph neural network-based fake new detection models using data-centric AI.

The code here is mostly taken from [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews), and modified for experimenting with the models, slice discovery, data removal, etc. See the original repo for details about the original code.

Original paper: SIGIR'21 ([PDF](https://arxiv.org/pdf/2104.12259.pdf))
```bibtex
@inproceedings{dou2021user,
  title={User Preference-aware Fake News Detection},
  author={Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

(Apologies for any strangeness in the code, I was learning Python when I started this project, and I wasn't expecting to be uploading it for someone else. I fixed up and documented everything I wrote as best I could. A lot of this is probably pretty inefficient too, sorry about that.)

## Installation Steps
This project was done in a Windows environment on a system with an NVIDIA GPU, so that's the setup these instructions are for.

1. The GNN models use NVIDIA's CUDA to utilize your GPU for increased speed. I used CUDA 11.7; the installer for CUDA can be downloaded from [here](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local).

    This code requires `Python>=3.6`, `PyTorch>=1.6`, and `PyTorch-Geometric>=1.6.1` to run. PyTorch must be installed with support for the your version of CUDA in order to use your GPU.

2. PyTorch can be installed through pip with this command. See [PyTorch's installation instructions](https://pytorch.org/get-started/locally/).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

3. PyTorch-Geometric and it's dependencies can be installed through pip with this command. See [PyG's installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
pip install torch_geometric==2.2.0 torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

4. [Domino](https://github.com/HazyResearch/domino/tree/main), a slice discovery method built for image datasets, was used in attempts at slice discovery. This can be installed through pip with this command. See [Domino installation instructions](https://domino-slice.readthedocs.io/en/latest/intro.html).
```bash
pip install domino
```

5. I had some issues with Domino and one of its dependencies it installs with, [meerkat](https://github.com/HazyResearch/meerkat), for unknown reasons. These are the changes I made that fixed them.

- Line 21 of *`domino/_embed/__init__`* should be changed to:
```python
def infer_modality(col: mk.Column):
```
- Line 26 of *`meerkat/constants.py`* should be changed to:
```python
assert os.path.abspath(__file__).endswith("constants.py"), (
```
- Line 174 of *`domino/gui.py`* should be changed to:
```python
mk.config.display.max_rows = page_size
```
- Line 180 of *`domino/gui.py`* should be changed to:
```python
dp[
```
6. Install additional requirements from requirements.txt:
