# Introduction

This repository is the code for our article [Driving Behavior Explanation with Multi-level Fusion](https://arxiv.org/abs/2012.04983), accepted at NeurIPS Workshop ML4AD 2020. It was built for `Python 3.7` using `PyTorch 1.3.1` and `bootstrap.pytorch 0.0.13` (see [this repo](https://github.com/Cadene/bootstrap.pytorch)).

# Training on HDD

To download the data, please go to [this website](https://usa.honda-ri.com/HDD). Then, make sure you have the data aranged in this form:

```
/HDD
	/EAF_parsing
	/release_2019_07_08
	/release_2019_07_25
	train.txt
	val.txt
```


The training is carried for 70K iterations on the `train` split. Then the performance is computed on the `val` split. Running any of the following scripts will first check if data preprocessing is done. If not, it will run the preprocessings before training the model.

To train Beef, simply run:
```
python -m bootstrap.run -o options/hdd_beef.yaml --dataset.dir_data /directory/to/hdd/folder
```

To run our multi-task baseline, run:
```
python -m bootstrap.run -o options/hdd_baseline_multitask.yaml --dataset.dir_data /directory/to/hdd/folder
```

To train the driver only model , run:
```
python -m bootstrap.run -o options/hdd_driver_only.yaml --dataset.dir_data /directory/to/hdd/folder
```

As we use `bootstrap.pytorch`, the command line overrides the `.yaml` option file. Thus, you can simply change the option files and put your dataset directory directly in there.

# Training on BDD-X

Please refer to [this repository](https://github.com/JinkyuKimUCB/BDD-X-dataset) for data downloading and preprocessing.

Then, you can train the driver using:
```
python -m bootstrap.run -o options/bdd_driver.yaml --dataset.dir_data /directory/to/bddx/folder
```

To train the captioning model based on BEEF, use:
```
python -m bootstrap.run -o options/bdd_caption.yaml --dataset.dir_data /directory/to/bddx/folder
```

# Citation

If you use our code and/or our article, you can cite us using:

```
@article{beef2020,
  author    = {Hedi Ben{-}Younes and
               {\'{E}}loi Zablocki and
               Patrick P{\'{e}}rez and
               Matthieu Cord},
  title     = {Driving Behavior Explanation with Multi-level Fusion},
  journal   = {NeurIPS Workshop on Machine Learning for Autonomous Driving (ML4AD)},
  year      = {2020}
}
```

