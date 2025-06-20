# State-Constrained Offline Reinforcment Learning

Pytorch implementation of State-constrained Deep Q-learning (StaCQ). The paper can be found [here](https://openreview.net/forum?id=KcR8ykFlHA).

StaCQ is an offline deep learning method that uses a learned formulation of state rechability to find higher performing policies than its batch-constrained counterparts.



### Usage

To recreate the results in the paper:
1. Train the forward and inverse models by running
   `Train_forward.py` and `Train_inverse.py`
   with the correct dataset selected.
   The pre-trained models can be found in the Models folder
2. Find the reachable states by running
   `Reachability_ablations.py`
   with `norm = 'linf', load_minmax = False, delta_X = 0.1` and the desired dataset selected.
3. Train Actor and critic by running
   `StaCQ_multistep_test_LOOP.py`

### bibtex
```
@article{hepburn2025state,
  title={State-Constrained Offline Reinforcement Learning},
  author={Hepburn, Charles A and Jin, Yue and Montana, Giovanni},
  journal={Transactions on Machine Learning Research},
  url={https://openreview.net/forum?id=KcR8ykFlHA}
  year={2025}
}
```
