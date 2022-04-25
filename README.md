# Modeling label space interactions using box embeddings

This is the official implementation for the paper [Modeling label space interactions using box embeddings](https://openreview.net/forum?id=tyTH9kOxcvh).

# Setup 

## Install the requirements

```
pip install -r requirements.txt
```

## Download data

Execute `download_data.sh`.

## Weights and Biases account

Since there are 12 datasets, 8 models and 10 runs with different 
random seeds for each dataset-model pair (960 runs in total), we recommend using
Weights and Biases server to log all the metrics.


1. Create a [wandb](https://docs.wandb.ai/quickstart#1.-set-up-wandb) account. It is free! 
2. Create a new wandb project. 
3. Login to wandb on your machine:

```commandline
wandb login
```

# Reproducing the metrics reported in the paper

1. Edit `run.sh` to use your wandb username and project name in `wandb_entity` and `wandb_project`, respectively.
2. Execute `run.sh` for different dataset and model settings.
3. Query the Weights and Biases server for the results. 
See the section below describing the methods to query the results.

See [official project page](https://wandb.ai/box-mlc/box-mlc-iclr-2022?workspace=user-dhruveshpate) see the runs used to report the results in the paper. 


# Tuning your own hyper-parameters
You can tune your own hyper-parameters for any model using the [wandb sweeps](https://docs.wandb.ai/guides/sweeps).

1. Create a model config file (jsonnet). See `model_configs` directory for examples.
2. Create a sweep config. See `example_sweep_configs` folder for example sweep configs, and [wandb docs](https://docs.wandb.ai/guides/sweeps/configuration) for further details.
3. [Create a sweep](https://docs.wandb.ai/guides/sweeps/quickstart#3.-initialize-a-sweep) by executing `wandb sweep path-to-sweep-config.yaml`.
4. [Start agents](https://docs.wandb.ai/guides/sweeps/quickstart#4.-launch-agent-s) for the sweep by executing `wandb agent <USERNAME/PROJECTNAME/SWEEPID>`.
5. Check the progress using the dashboard at `https://wandb.ai/USERNAME/PROJECT/sweeps/SWEEPID`.


# Querying the results
There are following three ways to see the results.


1. Use the command line utility called `wandb-utils` installed using `pip install wandb-utils==0.1.2`. Once installed, execute the following command:
```commandline
wandb-utils -e box-mlc -p box-mlc-iclr-2022 all-data \
filter-df --pd-eval "test_CMAP=rmax(df.test_MAP_max_n, df.test_MAP_min_n)" \
filter-df --pd-eval "_model=df.tags.str.extract(r'model@([^\|]+)',expand=False)" \
filter-df --pd-eval "_dataset=df.tags.str.extract(r'dataset@([^\|]+)',expand=False)" \
filter-df --pd-eval "df.groupby(['_model', '_dataset'], as_index=False).mean()" \
filter-df -f test_MAP -f test_CMAP -f test_constraint_violation  -f _model -f _dataset \
print
```

To use this for your own project, replace `box-mlc` and `box-mlc-iclr-2022` with your own 
username and project name.

2. Use wandb's python [api](https://docs.wandb.ai/guides/track/public-api-guide) directly.

3. Use the wandb [dashboard](https://wandb.ai/box-mlc/box-mlc-iclr-2022?workspace=user-dhruveshpate)


# NaN issue in MAP

There is a bug in sklearn that makes `mean_average_precision` return NaN when there are not true positives. Apply the one line fix mentioned in [this PR](https://github.com/scikit-learn/scikit-learn/pull/19085/files).
This fix has not been merged yet. Hence one has to patch this manually. Even after fixing this, if the dataset has a lot of instances where there are no true labels, you might need to disable the showing of the warning multiple times. For this, you can do the following based on python [docs](https://docs.python.org/3/library/warnings.html#warning-filter):

```
export PYTHONWARNINGS=once:::sklearn.metrics[.*]
```




# Cite
```
@inproceedings{
patel2022modeling,
title={Modeling Label Space Interactions in Multi-label Classification using Box Embeddings},
author={Dhruvesh Patel and Pavitra Dangati and Jay-Yoon Lee and Michael Boratko and Andrew McCallum},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=tyTH9kOxcvh}
}

```