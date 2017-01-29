# Tensorflow 0.x => 1.0 Migration Guide

1.0.0rc0 정식 출시에따라 겪게되는 맨붕 사태를 정리해 봤습니다.

## Common
### 초기화 : initialize_all_variables => global_variables_initializer or local_variables_initializer
2017-3-2 삭제 예정. 아직은 경고만 뜹니다만 미리 변경을 추천드립니다.

**v0.x 일반적인 코드**
```python
saver = tf.train.Saver(tf.all_variables())
```

**v1.0.0rc0**
```python
saver = tf.train.Saver(tf.global_variables())
```

## Saver
### all_variables => global_variables

**v0.x 일반적인 코드**
```python
sess.run(tf.initialize_all_variables())
```

**v1.0.0rc0**
```python
sess.run(tf.global_variables_initializer())
```
또는 
```python
sess.run(tf.local_variables_initializer())
```
### write_version tf.train.SaverDef.V1 => tf.train.SaverDef.V2
저장 포멧 변경으로 하위 호환이 사라짐. 하위 호환이 필요없다면 무시 가능

**v0.x 일반적인 코드**
```python
tf.train.Saver(max_to_keep=200)
```

**호환이 필요하다면 다음과 같이 해야함(미지정 시 기본 포멧이 V2로 저장됨)**
```python
tf.train.Saver(max_to_keep=200, write_version=tf.train.SaverDef.V1) 
```

## Summary
전체적으로 summery 관련 함수들이 tf에서 tf.summery로 이동되고 Summery prefix가 삭제되는 형태의 단순 이름 변경.

### SummaryWriter => FileWriter
단순히 이름만 변경됨.

**v0.x 일반적인 코드**
```python
train_writer = tf.train.SummaryWriter('log/', sess.graph)
```

**v1.0.0rc0**
```python
train_writer = tf.summary.FileWriter('log/', sess.graph)
```

### merge_all_summaries => summary.merge_all
단순히 이름만 변경됨.

**v0.x 일반적인 코드**
```python
merged = tf.merge_all_summaries()
```

**v1.0.0rc0**
```python
merged = tf.summary.merge_all()
```
### scalar_summary => summary.scalar
단순히 이름만 변경됨. 비슷한 류의 summary 함수들 모두 동일하게 
`*_summary => summary.*`

**v0.x 일반적인 코드**
```python
cost_summary = tf.scalar_summary('Cost', cost)
```

**v1.0.0rc0**
```python
cost_summary = tf.summary.scalar('Cost, cost)
```
## Optimizer
### softmax_cross_entropy_with_logits
Argument가 암시적 지정 방식을 더이상 지원하지 않습니다. 명시적 지정이 필요합니다.

**v0.x 일반적인 코드**
```python
cost = tf.nn.softmax_cross_entropy_with_logits(pred, y)
```

**v1.0.0rc0**
```python
cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
```

## RNN과 RNN에서 많이 쓰는 함수
### tf.nn.rnn_cell.*  => tf.contrib.rnn.*
패키지명 단순 이동.

**v0.x 일반적인 코드**
```python
encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)
```

**v1.0.0rc0**
```python
encoDecoCell = tf.contrib.rnn.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)
```

### tf.nn.seq2seq.* => tf.contrib.legacy_seq2seq.*
패키지명 단순 이동. legacy가 붙은걸 봐선 권장하는 방식이 아닐진데... 
더 이쁘게 업그레이드 하는 방법 아시는분 추천 부탁드립니다.

**v0.x 일반적인 코드**
```python
decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
     self.encoderInputs, 
     self.decoderInputs, 
     encoDecoCell,
     self.textData.getVocabularySize(),
     self.textData.getVocabularySize(),  
     embedding_size=self.args.embeddingSize,  
     output_projection=outputProjection.getWeights() if outputProjection else None,
     feed_previous=bool(self.args.test)  
 )
```

**v1.0.0rc0**
```python
decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
     self.encoderInputs, 
     self.decoderInputs, 
     encoDecoCell,
     self.textData.getVocabularySize(),
     self.textData.getVocabularySize(),  
     embedding_size=self.args.embeddingSize,  
     output_projection=outputProjection.getWeights() if outputProjection else None,
     feed_previous=bool(self.args.test)  
 )

```

### tf.split(split_dim, num_splits, value) –> tf.split(value, num_or_size_splits, axis)  !!!! 순서 바뀜 !!!!
잘돌아가던 rnn이 업그레이드 후 이상동작을 보인다면 95.135%는 여기가 문제입니다.

**v0.x 일반적인 코드**
```python
inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
```

**v1.0.0rc0**
```python
# inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, input_data))
inputs = tf.split(tf.nn.embedding_lookup(embedding, input_data), seq_length, 1)
```

### tf.concat(concat_dim, values) –> tf.concat_v2(values, axis) 
다행히 v2 함수가 따로 준비되어서 당분간 기존 함수로도 수정 없이 무난히 동작합니다.

**v0.x에 잘돌아가던 코드**
```python
output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
```

**v1.0.0rc0**
```python
output = tf.reshape(tf.concat_v2(outputs, 1), [-1, rnn_size])
```


---
## tensorflow.bloc 공식 발표에 나왔으나 아직 미정리 ㅠㅜ

* tf.pack / tf.unpack –> tf.stack / tf.unstack
    * v0.12: stack/unstack 추가, pack/unpack 은 stack/unstack 을 단순 wrapping
    * master: pack/unpack 에 deprecated 메세지 추가

* tf.concat(concat_dim, values) –> tf.concat_v2(values, axis)
    * v0.12: concat_v2 추가
    * master: concat 에 deprecated 메세지 추가

* tf.sparse_split(split_dim, num_split, sp_input) –> tf.sparse_split(sp_input, num_split, axis)
    * master: sparse_split(sp_input, num_split, axis) 로 변경

* tf.reverse(tensor, dims) –> tf.reverse(tensor, axis)
    * v0.12: reverse_v2(tensor, axis) 추가
    * master: reverse 가 바뀌고, v1.0 이후 reverse_v2 가 deprecated 될 예정

* tf.round –> banker’s rounding
    * v0.12: 파이썬의 banker’s rounding 으로 변경, 짝수로 반올림/내림

* dimension, dim, ~indices, ~dim, ~axes 키워드 파라미터 –> axis 로 통일
    * v0.12: 기존 파라미터와 axis 모두 유지, 향후 기존 파라미터 삭제 예정
    * tf.argmax: dimension –> axis
    * tf.argmin: dimension –> axis
    * tf.count_nonzero: reduction_indices –> axis
    * tf.expand_dims: dim –> axis
    * tf.reduce_all: reduction_indices –> axis
    * tf.reduce_any: reduction_indices –> axis
    * tf.reduce_join: reduction_indices –> axis
    * tf.reduce_logsumexp: reduction_indices –> axis
    * tf.reduce_max: reduction_indices –> axis
    * tf.reduce_mean: reduction_indices –> axis
    * tf.reduce_min: reduction_indices –> axis
    * tf.reduce_prod: reduction_indices –> axis
    * tf.reduce_sum: reduction_indices –> axis
    * tf.reverse_sequence: batch_dim –> batch_axis, seq_dim –> seq_axis
    * tf.sparse_concat: concat_dim –> axis
    * tf.sparse_reduce_sum: reduction_axes –> axis
    * tf.sparse_reduce_sum_sparse: reduction_axes –> axis
    * tf.sparse_split: split_dim –> axis

* tf.listdiff –> tf.setdiff1d
    * v0.12: setdiff1d 추가
    * master: listdiff 에 deprecated 메세지 추가

* tf.select –> tf.where
    * v0.12: where 추가
    * master: select 삭제

* tf.inv –> tf.reciprocal
    * v0.12: inv –> reciprocal 이름 변경

* tf.SparseTensor.shape –> tf.SparseTensor.dense_shape
    * master: shape –> dense_shape 이름변경

* tf.SparseTensorValue.shape –> tf.SparseTensorValue.dense_shape
    * master: shape –> dense_shape 이름 변경

* zeros_initializer –> 함수 리턴으로 변경
    * master: ones_initializer 와 동일한 리턴 형식으로 변경

* v0.12: TensorFlowEstimator() 가 삭제되었습니다. 대신 Estimator() 가 권장됩니다.
