import codecs
import json
import os

import tensorflow as tf

def noam_decay(hparams, global_step, learning_rate):
    """Defines the decay function described in https://arxiv.org/abs/1706.03762.
    """
    scale = tf.cast(learning_rate, tf.float32)
    step = tf.cast(global_step, tf.float32) + 1
    model_size = tf.cast(hparams.num_units, tf.float32)
    warmup_steps = tf.cast(hparams.warmup_steps, tf.float32)
    return scale * tf.pow(model_size, -0.5)  * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5))

def load_hparams(out_dir, overidded = None):
    hparams_file = os.path.join(out_dir,"hparams")
    print("# loading hparams from %s" % hparams_file)
    hparams_json = json.load(open(hparams_file))
    hparams = tf.contrib.training.HParams()
    for k,v in hparams_json.items():
        hparams.add_hparam(k,v)
    if overidded:
        for k,v in overidded.items():
            if k not in hparams_json:
                hparams.add_hparam(k,v)
            else:
                hparams.set_hparam(k,v)
    return hparams

def save_hparams(out_dir, hparams):
    """Save hparams."""
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    hparams_file = os.path.join(out_dir, "hparams")
    print("# saving hparams to %s" % hparams_file)
    with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
        f.write(hparams.to_json())

def get_total_param_num(params, threshold = 100000):
    total_parameters = 0
    #iterating over all variables
    for variable in params:  
        local_parameters=1
        shape = variable.get_shape()  #getting shape of a variable
        for i in shape:
            local_parameters*=i.value  #mutiplying dimension values
        if local_parameters >= threshold:
            print("variable {0} with parameter number {1}".format(variable, local_parameters))
        total_parameters+=local_parameters
    print('total parameter number',total_parameters) 
    return total_parameters

def get_config_proto(log_device_placement=True, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0, per_process_gpu_memory_fraction=0.95, allow_growth=True):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = allow_growth
    config_proto.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto
    
def create_hparams(flags):
    return tf.contrib.training.HParams(
        mode=flags.mode,
        max_len = flags.max_len,
        label_type = flags.label_type,
        batch_size = flags.batch_size,

        num_layers = flags.num_layers,
        num_units = flags.num_units,
        ffn_inner_dim = flags.ffn_inner_dim,
        num_heads = flags.num_heads,
        position_type = flags.position_type,
        l2_regularizer = flags.l2_regularizer,
        pooling = flags.pooling,
        fc_layer = flags.fc_layer,
        embedding_size = flags.embedding_size,
        pretrained_embedding_file = flags.pretrained_embedding_file,

        checkpoint_dir = flags.checkpoint_dir,
        learning_rate = flags.learning_rate,
        optimizer = flags.optimizer,
        max_gradient_norm = flags.max_gradient_norm,
        keep_prob = flags.keep_prob,
        max_to_keep = flags.max_to_keep,
        noam_decay = flags.noam_decay,
        warmup_steps = flags.warmup_steps
    )
