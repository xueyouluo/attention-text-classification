import codecs
import tensorflow as tf

UNK = '<unk>'
UNK_ID = 0
SOS = '<sos>'
SOS_ID = 1
EOS = '<eos>'
EOS_ID = 2

def read_vocab(vocab_file, check_vocab=True):
    """read vocab from file, one word per line
    """
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())

    if check_vocab:
        if vocab[UNK_ID] != UNK:
            vocab.insert(UNK_ID,UNK)
        if vocab[SOS_ID] != SOS:
            vocab.insert(SOS_ID,SOS)
        if vocab[EOS_ID] != EOS:
            vocab.insert(EOS_ID,EOS)

    word2id = {}
    for word in vocab:
        word2id[word] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word