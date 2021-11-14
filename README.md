# G3NN

This repo provides a pytorch implementation for the 4 instantiations of the flexible generative framework as described in the following paper:

[A Flexible Generative Framework for Graph-based Semi-supervised Learning](https://arxiv.org/abs/1905.10769)

[Jiaqi Ma](https://www.jiaqima.com/)\*, [Weijing Tang](https://sites.google.com/umich.edu/weijingtang/home)\*, [Ji Zhu](http://dept.stat.lsa.umich.edu/~jizhu/), and [Qiaozhu Mei](http://www-personal.umich.edu/~qmei/). NeurIPS 2019.

(\*: equal contribution)

## Requirements
See `environment.yml`. Run `conda torch_env create -f environment.yml` to install the required packages.

## Run the code
Example: `python main.py --model lsm_gcn --dataset cora`

## Cite
```
@inproceedings{ma2019flexible,
  title={A Flexible Generative Framework for Graph-based Semi-supervised Learning},
  author={Ma, Jiaqi and Tang, Weijing and Zhu, Ji and Mei, Qiaozhu},
  booktitle={Advances in Neural Information Processing Systems},
  pages={3276--3285},
  year={2019}
}
```
