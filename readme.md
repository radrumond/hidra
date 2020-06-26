# HIDRA: Head Initialization for Dynamic Robust Architectures

## Created by: Rafael Rego Drumond and Lukas Brinkmeyer

- Built on top of https://github.com/dragen1860/MAML-TensorFlow

## Credits and Details
Published in SIAM's SDM2020
- If you use this code, please cite the paper from MAML and our paper. Bibtex'es below. For more details, look at our paper.

Maml paper:
```
	@inproceedings{finn2017model,
	  title={Model-agnostic meta-learning for fast adaptation of deep networks},
	  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
	  booktitle={Proceedings of the 34th International Conference on Machine Learning-Volume 70},
	  pages={1126--1135},
	  year={2017},
	  organization={JMLR. org}
	}
```
Our paper:
```
@inproceedings{drumond2020hidra,
  title={HIDRA: Head Initialization across Dynamic targets for Robust Architectures},
  author={Drumond, Rafael Rego and Brinkmeyer, Lukas and Grabocka, Josif and Schmidt-Thieme, Lars},
  booktitle={Proceedings of the 2020 SIAM International Conference on Data Mining},
  pages={397--405},
  year={2020},
  organization={SIAM}
}
```

## How to use it?

- First you must download omniglot or miniimagenet data sets.

-- For omniglot use: https://github.com/cbfinn/maml/blob/master/data/omniglot_resized/resize_images.py

-- For the latter use: https://github.com/yaoyao-liu/mini-imagenet-tools


### How to run it?


- REGULAR MAML with OMNIGLOT (5 classes)

```
python main.py --arch maml  --dataset omni --data_path /path/to/omniglot --learning_rate 0.4 --meta_step 0.001 --meta_batch 4 --meta_iters 60001  --name OMNI_MAML --min_classes 5 --max_classes 5 --train_inner_K 1 --test_inner_K 3 --eval_iters 256
```

- REGULAR MAML with MiniImageNet (5 classes)

```
python main.py --arch maml  --dataset mini --data_path /path/to/MiniImageNet --learning_rate 0.4 --meta_step 0.001 --meta_batch 4 --meta_iters 60001  --name MINI_MAML_5 --min_classes 5 --max_classes 5 --train_inner_K 5 --test_inner_K 10 --eval_iters 256
```

- HIDRA with OMNIGLOT (5 classes)

```
python main.py --arch hidra  --dataset omni --data_path /path/to/omniglot --learning_rate 0.4 --meta_step 0.001 --meta_batch 4 --meta_iters 60001  --name OMNI_HIDRA_5 --min_classes 5 --max_classes 5 --train_inner_K 1 --test_inner_K 3 --eval_iters 256
```
-- or a lower learning rate

```
python main.py --arch hidra  --dataset omni --data_path /path/to/omniglot --learning_rate 0.01 --meta_step 0.001 --meta_batch 4 --meta_iters 60001  --name OMNI_HIDRA_5_SMALL_lr --min_classes 5 --max_classes 5 --train_inner_K 1 --test_inner_K 3 --eval_iters 256
```

- HIDRA with MiniImageNet (5 classes)

```
python main.py --arch hidra  --dataset mini --data_path /path/to/MiniImageNet --learning_rate 0.4 --meta_step 0.001 --meta_batch 4 --meta_iters 60001  --name MINI_HIDRA_5 --min_classes 5 --max_classes 5 --train_inner_K 5 --test_inner_K 10 --eval_iters 256
```

- For testing, just add the ```--test``` flag
- To continue training or for testing, you must provide the step number as ```--step X``` (where X is your checkpoint number on  model_checkpoint/NAME folder)

## Python Requirements

- tensorflow=1.14
- opencv-python
- pillow
- numpy
- json
