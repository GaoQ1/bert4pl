from absl import app, flags, logging
from glob import glob

import torch
import os
import copy
import torch.nn.functional as F
import pytorch_lightning as pl
import utils_seq2seq

from tokenization_unilm import UnilmTokenizer
from modeling_unilm import UnilmForSeq2SeqDecode, UnilmConfig

from torchnlp.random import set_seed

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('max_position_embeddings', 512, 'max position embeddings')
flags.DEFINE_integer('beam_size', 1, 'Beam size for searching')
flags.DEFINE_integer('min_len', None, '')

flags.DEFINE_integer('max_seq_length', 512, 'The maximum total input sequence length')
flags.DEFINE_integer('max_tgt_length', 128, 'maximum length of target sequence')

flags.DEFINE_float('length_penalty', 0.0, 'Length penalty for beam search')

flags.DEFINE_string('model_name_or_path', None, 'Path to pre-trained model or shortcut name selected in the list: ')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '')

FLAGS = flags.FLAGS

class QuestionAnswerGeneration(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.config = UnilmConfig.from_pretrained(
            FLAGS.model_name_or_path,
            max_position_embeddings=FLAGS.max_position_embeddings
        )

        self.tokenizer = UnilmTokenizer.from_pretrained(
            FLAGS.model_name_or_path
        )

        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

        self.model = UnilmForSeq2SeqDecode.from_pretrained(
            FLAGS.model_name_or_path,
            config=self.config,
            mask_word_id=mask_word_id,
            eos_id=eos_word_ids,
            sos_id=sos_word_id,
            search_beam_size=FLAGS.beam_size, 
            length_penalty=FLAGS.length_penalty,
            min_len=FLAGS.min_len
        )

    def forward(self, instance):
        input_ids, token_type_ids, position_ids, input_mask = instance

        traces = self.model(
            input_ids, 
            token_type_ids, 
            position_ids, 
            input_mask
        )
        return traces

def predict(model, text):
    tokenizer = model.tokenizer
    max_src_length = FLAGS.max_seq_length - 2 - FLAGS.max_tgt_length

    bi_uni_pipeline = utils_seq2seq.Preprocess4Seq2seqDecode(
        list(tokenizer.vocab.keys()), 
        tokenizer.convert_tokens_to_ids,
        FLAGS.max_seq_length, 
        max_tgt_length=FLAGS.max_tgt_length
    )

    text_tokenize = tokenizer.tokenize(text[:max_src_length])
    instances = [bi_uni_pipeline((text_tokenize, max_src_length))]
    instances = utils_seq2seq.batch_list_to_batch_tensors(instances)

    traces = model.forward(instances)

    if FLAGS.beam_size > 1:
        traces = {k: v.tolist() for k, v in traces.items()}
        output_ids = traces['pred_seq']
    else:
        output_ids = traces.tolist()
    

    output_lines = []
    
    for i in range(len(output_ids)):
        w_ids = output_ids[i]
        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
        output_tokens = []
        
        for t in output_buf:
            if t in ("[SEP]", "[PAD]"):
                break
            output_tokens.append(t)
        
        output_sequence = ''.join(output_tokens)
        output_lines.append(output_sequence)
    
    return output_lines


def main(argv):
    set_seed(FLAGS.seed)
    
    qa_checkpoints = list(sorted(glob(os.path.join(FLAGS.output_dir, "qa_*.ckpt"), recursive=True)))
    qa_model = QuestionAnswerGeneration.load_from_checkpoint(
        checkpoint_path= qa_checkpoints[-1]
    )
    qa_model.eval()
    qa_model.freeze()

    while True:
        text = input("qusetion is: ")
        result = predict(qa_model, text)
        print("answer is: ", result)

if __name__ == "__main__":
    app.run(main)
