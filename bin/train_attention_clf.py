import argparse
import json
import os
import pickle
import time

import numpy as np
import tensorflow as tf

from models.attention_classification import AttentionClassification
from utils.training_utils import create_hparams, save_hparams, load_hparams, get_config_proto
from utils.judger import Judger
from data.clf_dataset import CLFDataSet

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--mode", type=str, default='train', help="running mode: train | eval | inference")
    parser.add_argument("--restore", type='bool', nargs="?", const=True,
                      default=False,  help="Whether to restore arguments to continue training process")

    # Data settings
    parser.add_argument("--vocab_file", type=str, default=None, help="vocab file", required=True)
    parser.add_argument("--train_data_file", type=str, default=None, help="train data file", required=True)
    parser.add_argument("--eval_data_file", type=str, default=None, help="eval data file", required=True)
    parser.add_argument("--label_file", type=str, default=None, help="label file", required=True)
    parser.add_argument("--min_len", type=int, default=0, help="min lens of sentence or doc")
    parser.add_argument("--max_len", type=int, default=300, help="max lens of sentence or doc")
    parser.add_argument("--label_type", type=str, default='multi-class', help="label type: multi-class | multi-label")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")

    # Model settings
    parser.add_argument("--num_layers", type=int, default=4, help="number of attention layers")
    parser.add_argument("--num_units", type=int, default=256, help="hidden dim size")
    parser.add_argument("--ffn_inner_dim", type=int, default=1024, help="inner ffn dim")
    parser.add_argument("--pretrained_embedding_file", type=str, default=None, help="pretrained embedding file, should in glove format")
    parser.add_argument("--embedding_size", type=int, default=256, help="embedding size")
    parser.add_argument("--num_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--position_type", type=str, default='sine', help="position type: sine | learned")
    parser.add_argument("--l2_regularizer", type=float, default=0.005, help="l2 regularizer ratio")
    parser.add_argument("--pooling", type='bool', nargs="?", const=True,
                      default=False,  help="Whether to add pooling")
    parser.add_argument("--fc_layer", type=int, default=1, help="num of FC layers")

    # Training settings
    parser.add_argument("--num_train_epoch", type=int, default=10, help="training epoches")
    parser.add_argument("--steps_per_stats", type=int, default=10, help="steps to print stats")
    parser.add_argument("--steps_per_summary", type=int, default=10, help="steps to save summary")
    parser.add_argument("--steps_per_eval", type=int, default=10000, help="steps to save model")
    parser.add_argument("--checkpoint_dir", type=str, default='/tmp/attention_clf', help="checkpoint dir to save model",required=True)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate. Adam: 0.001 | 0.0001, Adagrad: 0.1")
    parser.add_argument("--noam_decay", type='bool', nargs="?", const=True,
                      default=True,  help="Whether to use noam decay")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="warmup steps")
    parser.add_argument("--optimizer", type=str, default='AdamOptimizer', help="AdamOptimizer | AdagradOptimizer")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm.")
    parser.add_argument("--keep_prob", type=float, default=0.7, help="drop out keep ratio for training")
    parser.add_argument("--max_to_keep", type=int, default=3, help="how many checkpoints to save")
    
def eval_for_train(eval_model, eval_sess, eval_data):
    jd = Judger(flags.label_file)
    eval_acc, batch_num = 0.0,0.0

    predicts = []
    for i,(sources,lengths,labels) in enumerate(eval_data.get_next()):
        accurary, predict, batch_size = eval_model.eval_one_batch(eval_sess, sources, lengths, labels)
        predicts.extend(predict.astype(np.int32))
        eval_acc += accurary
        batch_num += batch_size
    
    macro_f1, micro_f1 = jd.evaluation(flags.eval_data_file, predicts)
    accurary = eval_acc / batch_num
    return accurary,micro_f1, macro_f1

def train(flags):
    if flags.restore:
        try:
            old_flags = pickle.load(open(os.path.join(flags.checkpoint_dir,'flags.pkl'),'rb'))
        except:
            print("Can not restore old flags, training with new flags")
            old_flags = flags
        flags = old_flags

    train_data = CLFDataSet(
        vocab_file=flags.vocab_file, 
        label_file=flags.label_file, 
        data_file=flags.train_data_file, 
        batch_size=flags.batch_size,
        max_len=flags.max_len,
        min_len=flags.min_len,
        label_type=flags.label_type)

    eval_data = CLFDataSet(
        vocab_file=flags.vocab_file, 
        label_file=flags.label_file, 
        data_file=flags.eval_data_file, 
        batch_size=flags.batch_size,
        max_len=flags.max_len,
        min_len=0, #do not filter evaluation docs
        label_type=flags.label_type)

    hparams = create_hparams(flags)
    hparams.add_hparam("vocab_size", train_data.vocab_size)
    hparams.add_hparam("target_label_num", train_data.label_size)

    save_hparams(flags.checkpoint_dir, hparams)
    # save flags
    pickle.dump(flags, open(os.path.join(flags.checkpoint_dir,'flags.pkl'),'wb'))

    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    with train_graph.as_default():
        train_model = AttentionClassification(hparams)
        train_model.build()
        initializer = tf.global_variables_initializer()

    with eval_graph.as_default():
        eval_hparams = load_hparams(flags.checkpoint_dir,{"mode":'eval','checkpoint_dir':flags.checkpoint_dir+"/best_dev", "max_to_keep":1})
        eval_model = AttentionClassification(eval_hparams)
        eval_model.build()
    
    train_sess = tf.Session(graph=train_graph, config=get_config_proto(log_device_placement=False))
    eval_sess = tf.Session(graph=eval_graph, config=get_config_proto(log_device_placement=False))

    try:
        train_model.restore_model(train_sess)
    except:
        print("unable to restore model, initialize model with new params")
        train_model.init_model(train_sess, initializer=initializer)

    print("# Start to train with learning rate {0}, {1}".format(flags.learning_rate,time.ctime()))

    global_step = train_sess.run(train_model.global_step)
    best_acc = -1000000
    for epoch in range(flags.num_train_epoch):
        checkpoint_total_loss,step_time, checkpoint_loss, iters, checkpoint_acc,batch_num = 0.0,0.0, 0.0, 0,0.0, 0
        for i,(sources,lengths,labels) in enumerate(train_data.get_next(shuffle=True)):
            start_time = time.time()
            add_summary = (global_step % flags.steps_per_summary == 0)
            acc,label_losses,loss,predict_count,global_step,batch_size = train_model.train_one_batch(train_sess, sources, lengths, labels, add_summary = add_summary)
            step_time += (time.time() - start_time)
            batch_num += batch_size
            checkpoint_loss += label_losses
            checkpoint_acc += acc
            iters += predict_count
            checkpoint_total_loss += loss

            
            if global_step == 0:
                continue

            if global_step % flags.steps_per_stats == 0:
                train_acc = checkpoint_acc/batch_num
                acc_summary = tf.Summary()
                acc_summary.value.add(tag='accuracy', simple_value = train_acc)
                train_model.summary_writer.add_summary(acc_summary, global_step=global_step)

                print(
                    "# Epoch %d  global step %d batch %d/%d lr %g "
                    "label_loss %.5f total loss %.5f accuracy %.3f wps %.2f step time %.2fs" %
                    (epoch+1, global_step,i+1,train_data.num_batches, train_model.learning_rate.eval(session=train_sess),
                    checkpoint_loss/flags.steps_per_stats, checkpoint_total_loss/flags.steps_per_stats ,train_acc, iters/step_time, step_time/flags.steps_per_stats))
                checkpoint_total_loss, step_time, checkpoint_loss, iters, checkpoint_acc, batch_num = 0.0, 0.0, 0.0, 0, 0.0, 0

            if global_step % flags.steps_per_eval == 0:
                print("# global step {0}, eval model at {1}".format(global_step, time.ctime()))
                checkpoint_path  = train_model.save_model(train_sess)
                eval_model.saver.restore(eval_sess, checkpoint_path)
                acc, micro_f1, macro_f1 = eval_for_train(eval_model, eval_sess, eval_data)
                score = (macro_f1 + micro_f1) / 2
                print("##########")
                print("# Eval accuracy {0}, micro f1 {1}, macro f1 {2}".format(acc,micro_f1,macro_f1))
                print("# eval acc = {0}, best eval acc = {1}".format(score, best_acc))
                print("##########")
                if score > best_acc:
                    eval_model.save_model(eval_sess)
                    best_acc = score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    if flags.mode == 'train':
        train(flags)
    elif flags.mode == 'inference':
        pass