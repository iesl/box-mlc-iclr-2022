// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

// model and data specific variable
// dataset variables
local dataset_name = std.parseJson(std.extVar('dataset_name'));
//local dataset_name = 'expr_fun';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
//local ff_linear_layers=2;
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
//local ff_weight_decay = 0;


local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
{
  type: 'train_test_log_to_wandb',
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  validation_dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  evaluate_on_test: true,
  data_loader: {
    shuffle: true,
    batch_size: 4,
  },
  trainer: {
    num_epochs: if test == '1' then 3 else 200,
    patience: 8,
    validation_metric: '+MAP',
    cuda_device: std.parseInt(cuda_device),
    optimizer: {
      lr: 0.001,
      weight_decay: ff_weight_decay,
      type: 'adamw',
    },
    checkpointer: {
      type: 'default',
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      'track_epoch_callback',
      {
        type: 'wandb_allennlp',
        sub_callbacks: [
          {
            type: 'log_best_validation_metrics',
            priority: 100,
          },
        ],
        watch_model: false,
        should_log_parameter_statistics: false,
        save_model_archive: false,

      },
    ],
  },
  model: {
    debug_level: 0,
    type: 'baseline',
    scorer: 'hyperbolic',
    feedforward: {
      input_dim: num_input_features,
      num_layers: ff_linear_layers,
      hidden_dims: [ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [label_space_dim],
      activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + ['linear']),
    },
    hierarchy: {
      type: 'networkx-edgelist',
      filepath: data_dir + '/' + dataset_metadata.dir_name + '/' + 'hierarchy_tc.edgelist',
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*_feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
    constraint_violation: {
      hierarchy_reader: {
        type: 'networkx-edgelist',
        filepath: data_dir + '/' + dataset_metadata.dir_name + '/' + 'hierarchy_tc.edgelist',
      },
    },
    binary_nll_loss: true,
  },
}
