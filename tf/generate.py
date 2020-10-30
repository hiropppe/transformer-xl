import numpy as np
import model
import sentencepiece as spm
import tensorflow as tf
import time

from absl import flags
from gpu_utils import assign_to_gpu



flags.DEFINE_string("model_dir", default='./EXP-natsume/',
                    help="Estimator model_dir.")

flags.DEFINE_string("eval_ckpt_path", None, './EXP-natsume/')

flags.DEFINE_string("spm_file", '/data/txl/data/natsume/natsume.model', '')
flags.DEFINE_integer("num_generate", 30, '')

flags.DEFINE_bool("mcts", False, '')
flags.DEFINE_string("mod_reward", None, '')

flags.DEFINE_string("start_string", '今日は', '')

# Model config
flags.DEFINE_integer("tgt_len", default=1,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=640,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=True,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=400,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=16,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=410,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=410,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=41,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=2100,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.0,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=True,
                  help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=True,
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


def graph_fn(n_token):
    tower_mems = []

    with tf.device(assign_to_gpu(0, "/gpu:0")), \
            tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        inp_ph = tf.placeholder(tf.int32, [1, None])

        mems_i = [tf.placeholder(tf.float32, [FLAGS.mem_len, 1, FLAGS.d_model])
                  for _ in range(FLAGS.n_layer)]

        graph = single_core_graph(
            n_token=n_token,
            cutoffs=[],
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

    return graph, inp_ph, tower_mems, tower_mems_np


def apply_temperature(distribution, temperature=1.0):
    logits = np.log(distribution)
    logits = logits * temperature
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


class PolicyNet(object):

    def __init__(self, spm_file, action_fn, model_dir, ckpt_path=None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_file)
        self.n_token = self.sp.get_piece_size()

        self.actions, self.inp_ph, tower_mems, tower_mems_np = action_fn(self.n_token)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

        if ckpt_path is None:
            ckpt_path = tf.train.latest_checkpoint(model_dir)
        else:
            ckpt_path = ckpt_path

        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_path)

        self.feed_dict = {}
        for m, m_np in zip(tower_mems[0], tower_mems_np[0]):
            self.feed_dict[m] = m_np

    def __del__(self):
        if self.sess:
            self.sess.close()

    def encode_as_ids(self, string):
        return self.sp.encode_as_ids(string)

    def id_to_token(self, token_id):
        return self.sp.id_to_piece(token_id)

    def forward(self, token_ids, temperature=0.67):
        self.feed_dict[self.inp_ph] = np.expand_dims(token_ids, 0)
        fetched = self.sess.run([self.actions], feed_dict=self.feed_dict)
        probs = fetched[0][-1, -1]
        probs = apply_temperature(probs, temperature)
        return probs

    def generate_example(self, start_string='吾輩', n=10, temperature=0.67, greedy=False):
        start_ids = self.sp.encode_as_ids(start_string)
        ids = []
        ids.extend(start_ids)

        seq = [(i, 100.0) for i in ids]
        for i in range(n):
            self.feed_dict[self.inp_ph] = np.expand_dims(ids, 0)
            fetched = self.sess.run([self.actions], feed_dict=self.feed_dict)
            probs = fetched[0][-1, -1]
            if greedy:
                token_id = np.argmax(probs)
            else:
                probs = apply_temperature(probs, temperature)
                token_id = np.random.choice(self.n_token, 1, p=probs)
            seq.append((token_id, probs[token_id]))
            ids.append(int(token_id))

        return seq

    def beam_example(self, start_string='吾輩', n=10, k=10):
        from tqdm import tqdm
        start_ids = self.sp.encode_as_ids(start_string)

        top_k_sents = [[start_ids, .0]]
        for i in tqdm(range(n)):
            candidates = []
            for i, (seq, seq_score) in enumerate(top_k_sents):
                self.feed_dict[self.inp_ph] = np.expand_dims(seq, 0)
                fetched = self.sess.run([self.actions], feed_dict=self.feed_dict)
                probs = fetched[0][-1, -1]
                top_k = np.argpartition(probs, -k)[-k:]
                top_k_probs = probs[top_k]
                for token_id, prob in zip(top_k, top_k_probs):
                    each_candidate = [seq + [int(token_id)], seq_score + prob]
                    candidates.append(each_candidate)
            top_k_sents = sorted(candidates, key=lambda item: item[1], reverse=True)[:k]

        return [(' '.join([self.sp.id_to_piece(idx) for idx in seq[0]]), seq[1]) for seq in top_k_sents]


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    n = 10000
    max_depth = 100

    start_string = FLAGS.start_string

    lang_model = PolicyNet(FLAGS.spm_file, graph_fn, FLAGS.model_dir)

    start = time.time()
    seq = lang_model.generate_example(start_string=start_string, n=max_depth, greedy=True)
    elapsed = time.time() - start
    print('Greedy:')
    print('  Text: {:s}'.format(' '.join(lang_model.id_to_token(int(token[0])) for token in seq)))
    print('  IDs: {:s}'.format(str([token[0] for token in seq])))
    print('  Policy: {:s}'.format(str([token[1] for token in seq])))
    print('  Elapsed {:.4f} sec ({:.4f} ms/token)'.format(elapsed, elapsed*1000/max_depth))

    start = time.time()
    top_k_seq = lang_model.beam_example(start_string=start_string, n=max_depth, k=10)
    elapsed = time.time() - start
    print('Beam Search:')
    print('  TopK sents:')
    for k, (text, score) in enumerate(top_k_seq):
        print('    {:d}: {:s} ({:.4f})'.format(k, text, score))
    print('  Elapsed {:.4f} sec'.format(elapsed))

    if not FLAGS.mcts:
        return

    mod_reward = FLAGS.mod_reward

    from importlib import machinery
    from mcts import NLGGame, State, mcts_uct

    mod_reward = machinery.SourceFileLoader('mod_reward', mod_reward).load_module()

    game = NLGGame(max_depth=max_depth,
                   policy=lang_model.forward,
                   id_to_token=lang_model.id_to_token)

    start_ids = lang_model.encode_as_ids(start_string)

    current_state = State(actions=[start_ids[0]], reward_func=mod_reward.eval)
    if len(start_ids) > 1:
        for token_id in start_ids[1:]:
            current_state = game.result(current_state, token_id)

    start = time.time()
    best_sequence = mcts_uct(game, current_state, n, report_every=100)
    elapsed = time.time() - start
    print('Final Best Sequence:')
    print('  Start: {:s}'.format(start_string))
    print('  Start Ids: {:s}'.format(str(start_ids)))
    print('  Text: {:s}'.format(' '.join(lang_model.id_to_token(token_id) for token_id in best_sequence[-1].state.actions)))
    print('  IDs: ' + str(best_sequence[-1].state.actions))
    print('  Number of simulations: {:d}'.format(best_sequence[0].visits))
    print('  Policy: {:s}'.format('->'.join([str(node.P)[:5] for node in best_sequence])))
    print('  Visits: {:s}'.format('->'.join([str(node.visits) for node in best_sequence])))
    print('  Action Value: {:s}'.format('->'.join([str(node.value/node.visits)[:5] for node in best_sequence])))
    print('  Elapsed(sec): total {:.4f} sec {:.4f} ms/search'.format(elapsed, elapsed*1000/n))


if __name__ == "__main__":
    tf.app.run()
