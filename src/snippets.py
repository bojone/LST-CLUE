#! -*- coding: utf-8 -*-
# CLUE评测
# 模型配置文件

import os
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import AdaFactor
from bert4keras.optimizers import extend_with_gradient_accumulation

# 通用参数
data_path = '/root/clue/datasets/'
learning_rate = 5e-3
pooling = 'first'

# 权重目录
if not os.path.exists('weights'):
    os.mkdir('weights')

# 输出目录
if not os.path.exists('results'):
    os.mkdir('results')

# 模型路径
config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class LST_BERT(BERT):
    """定义新的BERT类
    """
    def __init__(self, expert_model=None, **kwargs):
        super(LST_BERT, self).__init__(**kwargs)
        self.expert_model = expert_model
        self.expert_layers = 0
        while True:
            try:
                n = 'Expert-Transformer-%d-FeedForward-Norm' % self.expert_layers
                expert_model.get_layer(n)
                self.expert_layers += 1
            except:
                break
        self.skip = self.expert_layers // self.num_hidden_layers

    def get_inputs(self):
        return self.expert_model.inputs

    def apply_embeddings(self, inputs):
        x = self.expert_model.get_layer('Expert-Embedding-Dropout').output
        x = keras.layers.Dense(self.hidden_size, use_bias=False)(x)
        return x

    def apply_main_layers(self, inputs, index):
        x = super(LST_BERT, self).apply_main_layers(inputs, index)
        n = (index + 1) * self.skip - 1
        n = 'Expert-Transformer-%d-FeedForward-Norm' % n
        y = self.expert_model.get_layer(n).output
        y = keras.layers.Dense(self.hidden_size, use_bias=False)(y)
        return keras.layers.Add()([x, y])

    def initializer(self, shape, dtype=None, order=2, gain=1.0):
        """使用DeepNorm的思想初始化
        """
        if shape[0] > 10000 or shape[0] < 10:
            hidden_size = shape[1]
        else:
            hidden_size = shape[0]
        gain *= self.num_hidden_layers**(-1. / order)
        stddev = 1.13684723 / hidden_size**0.5 * gain
        return K.truncated_normal(shape, stddev=stddev)


# 预训练模型
expert_model = build_transformer_model(
    config_path, checkpoint_path, prefix='Expert-'
)

for layer in expert_model.layers:
    layer.trainable = False

expert_model = keras.models.Model(expert_model.inputs, expert_model.outputs)

base = build_transformer_model(
    config_path,
    model=LST_BERT,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=512,
    expert_model=expert_model,
    return_keras_model=False
)

# 模型参数
last_layer = base.model.layers[-1].name

if pooling == 'first':
    pooling_layer = keras.layers.Lambda(lambda x: x[:, 0])
elif pooling == 'avg':
    pooling_layer = keras.layers.GlobalAveragePooling1D()
elif pooling == 'max':
    pooling_layer = keras.layers.GlobalMaxPooling1D()

# 优化器
AdaFactorG = extend_with_gradient_accumulation(AdaFactor, name='AdaFactorG')

optimizer = AdaFactor(
    learning_rate=learning_rate, beta1=0.9, min_dim_size_to_factor=10**6
)

optimizer2 = AdaFactorG(
    learning_rate=learning_rate,
    beta1=0.9,
    min_dim_size_to_factor=10**6,
    grad_accum_steps=2
)

optimizer4 = AdaFactorG(
    learning_rate=learning_rate,
    beta1=0.9,
    min_dim_size_to_factor=10**6,
    grad_accum_steps=4
)
