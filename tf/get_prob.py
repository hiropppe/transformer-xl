#! -*- coding:utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import model
import data_utils

from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for do_test evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("spm_file", None,
      help="Location of sentencepiece model")

# Model config
flags.DEFINE_integer("tgt_len", default=70,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")

# Input 
flags.DEFINE_string("sent", None,
      help="Sentence to calculate occurence score.")

FLAGS = flags.FLAGS


def get_model_fn(n_token, cutoffs):
  def model_fn(inp, tgt, mems, is_training):
    inp = tf.transpose(inp, [1, 0])

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    probs = model.decode(
        dec_inp=inp,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=False,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    # number of parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # format_str = '{{:<{0}s}}\t{{}}'.format(
    #     max([len(v.name) for v in tf.trainable_variables()]))
    # for v in tf.trainable_variables():
    #   tf.logging.info(format_str.format(v.name, v.get_shape()))
    return probs

  return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
  model_fn = get_model_fn(
      n_token=n_token,
      cutoffs=cutoffs)

  model_ret = model_fn(
      inp=inp,
      tgt=tgt,
      mems=mems,
      is_training=is_training)

  return model_ret


def get_prob(n_token, cutoffs, ps_device, spm_file, sent):
  import sentencepiece as spm
  sp = spm.SentencePieceProcessor()
  sp.Load(spm_file)
  sent_ids = sp.encode_as_ids(sent)
  print(' '.join('{:s}({:d})'.format(sp.id_to_piece(i), i) for i in sent_ids))
  if sent_ids[0] == 6:
    t = 2
  else:
    t = 1
  ids = []
  ids.extend(sent_ids[:t])

  tower_mems = []

  with tf.device(assign_to_gpu(0, ps_device)), \
      tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    inp_ph = tf.placeholder(tf.int32, [1, None])

    mems_i = [tf.placeholder(tf.float32, [FLAGS.mem_len, 1, FLAGS.d_model])
              for _ in range(FLAGS.n_layer)]

    prob = single_core_graph(
        n_token=n_token,
        cutoffs=cutoffs,
        is_training=False,
        inp=inp_ph,
        tgt=None,
        mems=mems_i)

    tower_mems.append(mems_i)

  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, 1, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]
      for core in range(1)
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path
    tf.logging.info("Evaluate {}".format(eval_ckpt_path))
    saver.restore(sess, eval_ckpt_path)

    fetches = [prob]

    feed_dict = {}
    for m, m_np in zip(tower_mems[0], tower_mems_np[0]):
      feed_dict[m] = m_np

    probs = [1.0] * t
    start = time.time()
    for i in range(t, len(sent_ids)):
        feed_dict[inp_ph] = np.expand_dims(ids, 0)
        fetched = sess.run(fetches, feed_dict=feed_dict)
        predictions = fetched[0]
        predictions = np.squeeze(predictions[-1], 0)
        prob = predictions[sent_ids[i]]
        probs.append(prob)
        ids.append(sent_ids[i])
        #print(' '.join([str(i) for i in ids]))

    print(' '.join('{:s}({:.7f})'.format(sp.id_to_piece(e[0]), e[1]) for e in zip(ids, probs)))
    print('occurence score={:.3f}'.format(np.sum(np.log(np.array(probs[t:]) + 1e-10))))
    print('elapsed={:.4f} ms'.format((time.time() - start)*1000))

def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]
  tf.logging.info("n_token {}".format(n_token))

  get_prob(n_token, cutoffs, "/gpu:0", FLAGS.spm_file, FLAGS.sent)


if __name__ == "__main__":
  tf.app.run()

