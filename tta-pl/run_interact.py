from absl import app, flags, logging
from glob import glob

import torch
import os
import copy
import torch.nn.functional as F
import pytorch_lightning as pl

from modeling_tta import TtaLMHeadModel
from transformers import AutoConfig, AutoTokenizer

from torchnlp.random import set_seed

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('batch_size', 5, '批量生成response, 然后经过MMI模型进行筛选')

flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('max_history_len', 5, 'dialogue history的最大长度')
flags.DEFINE_integer('max_gen_token_len', 25, '每个utterance的最大长度,超过指定长度则进行截断')
flags.DEFINE_integer('topk', 8, '最高k选1')

flags.DEFINE_float('repetition_penalty', 1.0, '重复惩罚参数，若生成的对话重复性较高，可适当提高该参数')
flags.DEFINE_float('temperature', 1.0, '生成的temperature')
flags.DEFINE_float('topp', 0, '最高积累概率')

flags.DEFINE_string('model_name_or_path', 'bert-base-chinese', '')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '')

FLAGS = flags.FLAGS


class ChitChat(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            FLAGS.model_name_or_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.model_name_or_path
        )

        self.model = TtaLMHeadModel.from_pretrained(
            FLAGS.model_name_or_path,
            config=self.config
        )

    def forward(self, input_ids, labels=None, output_all_encoded_layers=False):
        outputs = self.model(input_ids, labels=labels, output_all_encoded_layers=output_all_encoded_layers)
        return outputs

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict(model, text, history=[]):
    with torch.no_grad():
        tokenizer = model.tokenizer

        history.append(tokenizer.encode(text, add_special_tokens=False))
        input_ids = [tokenizer.cls_token_id]

        for history_utr in history[-FLAGS.max_history_len:]:
            input_ids.extend(history_utr)
            input_ids.append(tokenizer.sep_token_id)

        # 用于批量生成response，维度为(batch_size, token_len)
        input_ids = [copy.deepcopy(input_ids) for _ in range(FLAGS.batch_size)]
        curr_input_tensors = torch.tensor(input_ids, dtype=torch.long)

        generated = []
        finish_set = set()
        candidate_responses = []

        for _ in range(FLAGS.max_gen_token_len):
            outputs = model.forward(input_ids=curr_input_tensors)
            next_token_logits = outputs[0][:, -1, :]

            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for index in range(FLAGS.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= FLAGS.repetition_penalty
            next_token_logits = next_token_logits / FLAGS.temperature

            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            for next_token_logit in next_token_logits:
                next_token_logit[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=FLAGS.topk, top_p=FLAGS.topp)

            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # 判断是否有response生成了[SEP],将已生成了[SEP]的resposne进行标记
            for index, token_id in enumerate(next_token[:, 0]):
                if token_id == tokenizer.sep_token_id:
                    finish_set.add(index)
            
            # 检验是否所有的response均已生成[SEP]
            finish_flag = True  # 是否所有的response均已生成[SEP]的token
            for index in range(FLAGS.batch_size):
                if index not in finish_set:  # response批量生成未完成
                    finish_flag = False
                    break
            if finish_flag:
                break
            generated.append([token.item() for token in next_token[:, 0]])
            
            # 将新生成的token与原来的token进行拼接
            curr_input_tensors = torch.cat((curr_input_tensors, next_token), dim=-1)

        for batch_index in range(FLAGS.batch_size):
            response = []
            for token_index in range(len(generated)):
                if generated[token_index][batch_index] != tokenizer.sep_token_id:
                    response.append(generated[token_index][batch_index])
                else:
                    break
            candidate_responses.append(response)

        result_text = [tokenizer.convert_ids_to_tokens(candidate_response) for candidate_response in candidate_responses]

    return result_text


def main(argv):
    set_seed(FLAGS.seed)
    
    nlg_tta_checkpoints = list(sorted(glob(os.path.join(FLAGS.output_dir, "nlg_tta_*.ckpt"), recursive=True)))
    nlg_tta_model = ChitChat.load_from_checkpoint(
        checkpoint_path= nlg_tta_checkpoints[-1]
    )
    nlg_tta_model.eval()
    nlg_tta_model.freeze()

    history = []

    while True:
        text = input("user: ")
        result = predict(nlg_tta_model, text, history)

        print(result)

if __name__ == "__main__":
    app.run(main)
