import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad


def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=128,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results

# def convert_example(example, tokenizer, is_test=False):

#     input_ids = tokenizer.encode(example["text"])
#     input_ids = np.array(input_ids, dtype='int64')

#     if not is_test:
#         label = np.array(example["label"], dtype="int64")
#         return input_ids, label
#     else:
#         return input_ids
def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)