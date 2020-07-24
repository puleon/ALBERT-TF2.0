import tensorflow as tf
from albert import AlbertModel, AlbertConfig

max_seq_length = 43
checkpoint_file = '/home/leonid/MODELS/albert_mem_base/ctl_step_2800000.ckpt-14'
config_fname = '/home/leonid/MODELS/albert_mem_base/config.json'

input_word_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
input_mask = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
input_type_ids = tf.keras.layers.Input(
    shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
config = AlbertConfig.from_json_file(config_fname)
config.output_attentions = True
albert_layer = AlbertModel(config=config, output_attentions=True)
pooled_output, sequence_output, attention_weights = albert_layer(input_word_ids, input_mask, input_type_ids)
albert_model = tf.keras.Model(inputs=[input_word_ids,input_mask,input_type_ids],
                              outputs=[pooled_output, sequence_output, attention_weights])

checkpoint = tf.train.Checkpoint(model=albert_model)
checkpoint.restore(checkpoint_file).assert_existing_objects_matched()