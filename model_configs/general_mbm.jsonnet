// env variables
local root_dir = std.extVar('ROOT_DIR');
local data_dir = std.extVar('DATA_DIR');
local test = std.extVar('TEST');  // a test run with small dataset
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);
// model class specific variable
local dataset_name = std.parseJson(std.extVar('dataset_name'));
//local dataset_name = 'cellcycle_fun';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

//local gumbel_beta = std.parseJson(std.extVar('gumbel_beta'));
local gumbel_beta = 0.0001;
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
//local ff_dropout = std.parseJson(std.extVar('ff_dropout'));
local ff_dropout = 0;
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
//local ff_linear_layers=2;
local lr_encoder = std.parseJson(std.extVar('lr_encoder'));
//local lr_ratio = std.parseJson(std.extVar('lr_ratio'));
local lr_ratio = 10;
local lr_boxes = lr_ratio * lr_encoder;
//local ff_weight_decay = 0;
//local volume_temp = 1;
local volume_temp = std.parseJson(std.extVar('volume_temp'));
//local volume_temp = 0.1;
//local step_up = std.parseJson(std.extVar('step_up'));
//local box_space_dim = std.parseInt(std.toString(std.floor(ff_hidden / step_up)));
local box_space_dim = ff_hidden;
//local delta = std.parseJson(std.extVar('delta'));
local delta = 1e-5;
//local label_delta_init = std.parseJson(std.extVar('label_delta_init'));
local label_delta_init = 0.01;
//local box_weight_decay = std.parseJson(std.extVar('box_weight_decay'));
local box_weight_decay = 0;
//local final_activation = if std.parseJson(std.extVar('ff_same_final_activation')) then ff_activation else 'linear';
local final_activation = ff_activation;

local gain = if ff_activation == 'tanh' then 5 / 3 else 1;
{
  type: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  validation_dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

  model: {
    type: 'mbm',
    debug_level: 0,
    feedforward: {
      input_dim: num_input_features,
      num_layers: ff_linear_layers,
      //hidden_dims: [ff_hidden*step_up, ff_hidden],
      //activations: [ff_activation, ff_activation],
      //dropout: [ff_dropout, ff_dropout],
      hidden_dims: [ff_hidden for i in std.range(0, ff_linear_layers - 2)] + [box_space_dim],
      activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [final_activation]),
      dropout: ff_dropout,
      //dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
    },
    vec2box: {
      type: 'vec2box',
      box_factory: {
        type: 'box_factory',
        kwargs_dict: {
          delta: delta,
        },
        name: 'center_fixed_delta_from_vector',
      },
    },
    intersect: { type: 'gumbel', beta: gumbel_beta },
    volume: { type: 'bessel-approx', log_scale: true, beta: volume_temp, gumbel_beta: gumbel_beta },
    label_embeddings: {
      type: 'box-embedding-module',
      embedding_dim: box_space_dim,
    },
    constraint_violation: {
      hierarchy_reader: {
        type: 'networkx-edgelist',
        filepath: data_dir + '/' + dataset_metadata.dir_name + '/' + 'hierarchy_tc.edgelist',
      },
    },
    initializer: {
      regexes: [
        [@'.*linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform'})],
        [@'.*linear_layers.*bias', { type: 'zero' }],
        [
          '_label_embeddings.center.weight',
          {
            //gain: 1,
            type: 'uniform',
          },
        ],
        [
          '_label_embeddings.delta',
          {
            type: 'normal',
            mean: label_delta_init,
            std: label_delta_init / 2.0,
          },
        ],
      ],
    },

  },
  data_loader: {
    shuffle: true,
    batch_size: 4,
  },
  trainer: {
    num_epochs: if test == '1' then 20 else 200,
    patience: 8,
    validation_metric: '+MAP',
    cuda_device: std.parseInt(cuda_device),
    optimizer: {
      type: 'adamw',
      lr: lr_encoder,  //lr_box
      weight_decay: 0,
      parameter_groups: [
        [
          [
            '.*linear_layers.*weight',
            '.*linear_layers.*bias',
          ],
          {
            lr: lr_encoder,
            weight_decay: ff_weight_decay,
          },
        ],
        [
          [
            '.*label_embeddings.delta',
          ],
          {
            lr: lr_boxes*10,
            weight_decay: box_weight_decay,
          },
        ],
        [
          [
            '.*label_embeddings.center',
          ],
          {
            lr: lr_boxes,
            weight_decay: 0,
          },
        ],
      ],
    },
    callbacks: [
      "track_epoch_callback",
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
    checkpointer: {
      type: 'default',
      keep_most_recent_by_count: 1,
    },
  },
}
