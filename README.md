# A Variational Approach for Learning from Positive and Unlabeled Data

This repository is the official implementation of [A Variational Approach for Learning from Positive and Unlabeled Data](https://arxiv.org/abs/1906.00642). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and Evaluation

To repeat experiments in the paper, run the following commands:

```train
python run.py --dataset cifar10 --lam 0.03 --num_labeled 3000 --learning-rate 3e-5 --gpu <gpu_id>

python run.py --dataset fashionMNIST --lam 0.3 --num_labeled 3000 --learning-rate 3e-4 --gpu <gpu_id>

python run.py --dataset stl10 --lam 0.3 --num_labeled 2250 --learning-rate 1e-4 --gpu <gpu_id>

python run.py --dataset pageblocks --lam 0.0001 --num_labeled 100 --learning-rate 3e-4 --batch-size 100 --gpu <gpu_id>

python run.py --dataset grid --lam 0.1 --num_labeled 1000 --learning-rate 3e-4 --gpu <gpu_id>

python run.py --dataset avila --lam 0.1 --num_labeled 2000 --learning-rate 6e-4 --gpu <gpu_id>

```


## Results

Our model achieves the following performance (accuracy) on PU learing tasks of FashionMNIST, CIFAR-10 and STL-10:

| Model     | FashionMNIST |   CIFAR-10  |   STL-10    |
| --------- | ------------ | ----------- | ----------- |
| VPU       |    92.7%     |    89.5%    |    79.7%    |
| nnPU      |	 90.8%     |    85.6%    |	  78.3%    | 


where nnPU is the current state-of-the-art. For more details, please refer to Table 2 and 3 in the paper.


## Cite the paper

If you find this useful, please cite

```
@article{chen2019variational,
  title={A Variational Approach for Learning from Positive and Unlabeled Data},
  author={Chen, Hui and Liu, Fangqing and Wang, Yin and Zhao, Liyue and Wu, Hao},
  journal={arXiv preprint arXiv:1906.00642},
  year={2019}
}  
```
