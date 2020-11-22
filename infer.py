import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os
import re
import json
import string
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, AutoTokenizer

max_len = 256
configuration = BertConfig()  # default parameters and configuration for BERT

tokenizer = BertWordPieceTokenizer("phobert-base/vocab.txt", lowercase=True)

def preprocess(context, question):
    # Clean context, answer and question
    context = " ".join(str(context).split())
    question = " ".join(str(question).split())

    # Tokenize context
    tokenized_context = tokenizer.encode(context)

    # Tokenize question
    tokenized_question = tokenizer.encode(question)

    # Create inputs
    input_ids = tokenized_context.ids + tokenized_question.ids[1:]
    token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
        tokenized_question.ids[1:]
    )
    attention_mask = [1] * len(input_ids)

    # Pad and create attention masks.
    # Skip if truncation is needed
    padding_length = max_len - len(input_ids)

    if padding_length > 0:  # pad
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
    elif padding_length < 0:  # skip
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        token_type_ids = token_type_ids[:max_len]

    return input_ids, token_type_ids, attention_mask

def create_model():
    ## BERT encoder
    encoder = TFBertModel.from_pretrained("vinai/phobert-base")

    ## QA Model
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])

    if os.path.exists("./ckpt/checkpoint"):
        model.load_weights("./ckpt/checkpoint")

    return model

use_tpu = False
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()

elif len(tf.config.experimental.list_physical_devices('GPU')) > 1:
    strategy = tf.distribute.MirroredStrategy()
    # Create model
    with strategy.scope():
        model = create_model()
        
else:
    model = create_model()

model.summary()

def infer(context, question, threshold=0.1):
    input_ids, token_type_ids, attention_mask = preprocess(context, question)

    res = model.predict([[input_ids], [token_type_ids], [attention_mask]])

    idy, idx, idz = np.where(np.array(res) > threshold)
    index = np.array([idx, idy, idz])

    list_answer = []
    for x in set(sorted(idx)):
        ques_idx = np.where(index[0] == x)
        start_idx = np.where(index[1] == 0)
        end_idx = np.where(index[1] == 1)

        list_start = np.intersect1d(ques_idx, start_idx)
        list_end = np.intersect1d(ques_idx, end_idx)

        answers = []
        for pair in zip(list_start, list_end):
            idx_s, idx_e = index[2][pair[0]], index[2][pair[1]]
            prob_s = res[index[1][pair[0]]][index[0][pair[0]]][index[2][pair[0]]]
            prob_e = res[index[1][pair[1]]][index[0][pair[1]]][index[2][pair[1]]]
            prob = (prob_s + prob_e) / 2

            if idx_s == idx_e and idx_s == 0:
                answers.append(("", prob))
            else:
                str_dec = tokenizer.decode(input_ids[idx_s:idx_e+1])
                answers.append((str_dec, prob))

        list_answer.append(answers)

    return list_answer
