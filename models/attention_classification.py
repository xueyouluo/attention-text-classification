import os

import numpy as np
import tensorflow as tf

import opennmt as onmt

from utils.training_utils import get_total_param_num, noam_decay

class AttentionClassification(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def build(self):
        self.setup_placeholders()
        self.setup_embedding()
        self.setup_encode()
        self.setup_classification()
        if self.hparams.mode in ['train','eval']:
            self.setup_loss()
        if self.is_training:
            self.setup_train()
            self.setup_summary()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.hparams.max_to_keep)

    def init_model(self, sess, initializer=None):
        if initializer:
            sess.run(initializer)
        else:
            sess.run(tf.global_variables_initializer())

    def save_model(self, sess):
        return self.saver.save(sess, os.path.join(self.hparams.checkpoint_dir,
                            "model.ckpt"), global_step=self.global_step)

    def restore_model(self, sess, epoch = None):
        if epoch is None:
            self.saver.restore(sess, tf.train.latest_checkpoint(
                self.hparams.checkpoint_dir))
        else:
            self.saver.restore(
                sess, os.path.join(self.hparams.checkpoint_dir, "model.ckpt" + ("-%d" % epoch)))
        print("restored model")

    @property
    def is_training(self):
        return self.hparams.mode == 'train'

    def setup_placeholders(self):
        self.source_tokens = tf.placeholder(tf.int32, shape=[None, None], name='source_tokens')
        self.source_length = tf.placeholder(tf.int32, shape=[None], name='source_length')
        if self.hparams.mode in ['train','eval']:
            self.target_labels = tf.placeholder(tf.float32, shape=[None, None], name='target_labels')

        self.batch_size = tf.shape(self.source_tokens)[0]
        self.global_step = tf.train.get_or_create_global_step()

    def setup_embedding(self):
        with tf.variable_scope("embedding") as scope:
            self.embedding = tf.get_variable('embedding',shape=[self.hparams.vocab_size, self.hparams.embedding_size])

            self.inputs = tf.nn.embedding_lookup(self.embedding, self.source_tokens)
            if self.is_training:
                self.inputs = tf.nn.dropout(self.inputs, self.hparams.keep_prob)

    def setup_encode(self):
        with tf.variable_scope("encode", reuse=tf.AUTO_REUSE) as scope:
            if self.hparams.position_type == 'learned':
                position_encoder = onmt.layers.position.PositionEmbedder(maximum_position=self.hparams.max_len)
            elif self.hparams.position_type == 'sine':
                position_encoder = onmt.layers.position.SinusoidalPositionEncoder()
            else:
                raise ValueError("position type %s not supported" % self.hparams.position_type)

            attention_encoder = onmt.encoders.SelfAttentionEncoder(
                num_layers = self.hparams.num_layers,
                num_units = self.hparams.num_units,
                num_heads = self.hparams.num_heads,
                ffn_inner_dim = self.hparams.ffn_inner_dim,
                dropout = 1 - self.hparams.keep_prob,
                attention_dropout = 0.1, # these are default values, change it in case needed
                relu_dropout = 0.1,
                position_encoder = position_encoder
            )

            # attention outputs: [batch * length * num_units]
            # state: (batch * num_units) * num_layers. Note: they are mean values of outputs of each layer
            # sequence_length: [batch]
            self.attention_outputs, self.state, self.sequence_length = attention_encoder.encode(
                inputs = self.inputs,
                sequence_length = self.source_length,
                mode = tf.estimator.ModeKeys.TRAIN if self.is_training else tf.estimator.ModeKeys.PREDICT
            )

            self.predict_count = tf.reduce_sum(self.sequence_length)

    def setup_classification(self):
        # add l2 regularizer
        if self.hparams.l2_regularizer > 0.0:
            regularizer = tf.contrib.layers.l2_regularizer(self.hparams.l2_regularizer)
        else:
            regularizer = None

        with tf.variable_scope("classification", regularizer=regularizer) as scope:
            # choose the last output as final state
            inp = self.attention_outputs[:,-1]

            # add pooling info
            if self.hparams.pooling:
                max_pool = tf.reduce_max(self.attention_outputs, axis=1)
                mean_pool = tf.reduce_mean(self.attention_outputs, axis=1)
                # [batch * (3*num_units)]
                inp = tf.concat([inp,max_pool,mean_pool],axis=-1)

            # add FC layers
            for i in range(self.hparams.fc_layer):
                if self.is_training:
                    inp = tf.nn.dropout(inp, self.hparams.keep_prob)
                inp = tf.layers.dense(inp, self.hparams.ffn_inner_dim, activation=tf.nn.relu)
        
            if self.is_training:
                inp = tf.nn.dropout(inp, self.hparams.keep_prob)

            # final output layer
            self.label_outputs = tf.layers.dense(inp, self.hparams.target_label_num)

            if self.hparams.label_type == 'multi-class':
                self.label_predict = self.label_outputs
            elif self.hparams.label_type == 'multi-label':
                print('multi-label')
                self.label_predict = tf.sigmoid(self.label_outputs)
            else:
                raise ValueError("label type %s not supported" % self.hparams.label_type)

            if self.hparams.mode in ['train','eval']:
                if self.hparams.label_type == 'multi-class':
                    predict = tf.argmax(self.label_outputs,axis=-1)
                    predict = tf.one_hot(predict,self.hparams.target_label_num)
                else:
                    condition = tf.greater_equal(self.label_predict, 0.5)
                    predict = tf.where(condition,tf.ones_like(self.target_labels),tf.zeros_like(self.target_labels))
                self.accurary = tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(predict,self.target_labels),axis=1),tf.float32))
            self.predict = predict
            

    def setup_loss(self):
        with tf.name_scope("losses") as scope:
            # TODO: add weighted loss and focal loss
            if self.hparams.label_type == 'multi-class':
                label_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.label_outputs, labels=self.target_labels)
            else:
                label_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.label_outputs,labels=self.target_labels)

            self.label_losses = tf.reduce_sum(label_losses) / tf.to_float(self.batch_size)

            losses = self.label_losses
            tf.losses.add_loss(losses)
            self.losses = tf.losses.get_total_loss()

    def setup_train(self):
        # TODO: add learning rate decay
        self.learning_rate = tf.constant(self.hparams.learning_rate,dtype=tf.float32)
        if self.hparams.noam_decay:
            self.learning_rate = noam_decay(self.hparams, self.global_step, self.learning_rate)

        opt_fn = onmt.utils.optim.get_optimizer_class(self.hparams.optimizer)
        if self.hparams.optimizer == 'AdamOptimizer':
            # setting from https://arxiv.org/abs/1706.03762
            opt = opt_fn(self.learning_rate, beta1=0.9, beta2=0.98,  epsilon=1e-09)
        else:
            opt = opt_fn(self.learning_rate)

        params = tf.trainable_variables()
        get_total_param_num(params)

        gradients = tf.gradients(self.losses, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hparams.max_gradient_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.train_op = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

    def setup_summary(self):
        self.summary_writer = tf.summary.FileWriter(
            self.hparams.checkpoint_dir, tf.get_default_graph())
        tf.summary.scalar("train_loss", self.losses)
        tf.summary.scalar("learning_rate", self.learning_rate)
        tf.summary.scalar('gN',self.gradient_norm)
        tf.summary.scalar('pN',self.param_norm)
        self.summary_op = tf.summary.merge_all()

    def train_one_batch(self, sess, sources, lengths, labels, add_summary=False):
        feed_dict = {self.source_tokens:sources, self.source_length:lengths, self.target_labels:labels}
        _,accurary,label_losses,loss,predict_count,summary,global_step,batch_size = sess.run(
            [self.train_op, self.accurary,self.label_losses, self.losses,self.predict_count,self.summary_op,self.global_step,self.batch_size],
            feed_dict=feed_dict)
        if add_summary:
            self.summary_writer.add_summary(summary, global_step=global_step)
        return accurary,label_losses,loss,predict_count,global_step, batch_size

    def eval_one_batch(self, sess, sources, lengths, labels):
        feed_dict = {self.source_tokens:sources, self.source_length:lengths, self.target_labels:labels}
        accurary, predict, batch_size = sess.run([self.accurary, self.predict, self.batch_size], feed_dict=feed_dict)
        return accurary, predict, batch_size