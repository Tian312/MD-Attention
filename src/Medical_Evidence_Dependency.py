# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from run_bluebert.py in  ncbi_bluebert


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import combinations
import collections
import csv
import logging
import os,re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from bert import modeling
from bert import optimization
from bert import tokenization
from parser_config import Config
config=Config()

do_lower_case = False
do_train = False
do_eval = False
do_predict = True
use_tpu = False
learning_rate = 5e-5
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples to be a multiple of the batch size, because the TPU requires a fixed batch size. The alternative is to drop the last batch, which is bad because it means the entire output data won't be generated.
                                                                    """

class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()
    
    def get_examples_from_pico(self, words, tags,sent_id="TEST"):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class BlueBERTProcessor(DataProcessor):
    """Processor for the BLUE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
    
    def file_based_convert_examples_to_features(self,
        examples, label_list, max_seq_length, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""

        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            #if ex_index % 10000 == 0:
            #    tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()

    def file_based_input_fn_builder(self,input_file, seq_length, is_training,
                                drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn    
        
    #---------------processing input from csv file with header--------------------------#
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = tokenization.convert_to_unicode(line[1])
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                try:
                    label = tokenization.convert_to_unicode(line[2])
                except IndexError:
                    logging.exception(line)
                    exit(1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    
    
    #------------- processing input from PICO element recognition results ---------#

    def get_examples_from_pico(self, words, tags,sent_id="TEST"):
        return self._create_examples_from_pico(words,tags,sent_id) 
    
    def _text2relation(self, sent_id, term_ann, text_tagged,relation_ann=[]):
        text_tagged = re.sub("\s+\S+__I\-\w+__O","",text_tagged)
        text_tagged = re.sub(r"\s*\S+__B\-(\w+)__(T\d+)",r" @\1$__\2",text_tagged)
        lines=" ".join(text_tagged.split(r'[;\n]'))
        relation_list = []
        term_list = re.findall("__(T\d+)",text_tagged)
        if len(term_list) <1:
            return []
        comb = list(combinations(term_list, 2))
        for tup in comb:
            if re.search("\w\.$",text_tagged):
                text_tagged = re.sub("\.$", " .",text_tagged)
            sent_temp = text_tagged+" "

            others = term_list.copy()
            others.remove(tup[0])
            others.remove(tup[1])
             
            index = sent_id+"."+tup[0]+"."+tup[1]
            # restore all other terms to text
            for t in others:
                pattern = re.compile("\S+__"+t+" ")
                sent_temp = re.sub(pattern,term_ann[t]+" ",sent_temp)
            for t in tup:
                pattern = re.compile("__"+t)
                sent_temp = re.sub(pattern,"",sent_temp)
            if not re.search("Count\$|Observation\$", sent_temp) or re.search("Participant\$", sent_temp):
                continue
            if tup in relation_ann or (tup[1],tup[0]) in relation_ann:
                output = [index,sent_temp,"1"]
            else:
                output = [index, sent_temp,"0"] 
            relation_list.append(output) 
        return relation_list
            
    def _create_examples_from_pico(self, words, tags,sent_id="TEST"):
        """ INPUT: one sentence pico results """
        examples = []
        text_tagged = ""
        word_tagged=[]
        term_dict={}
        entity_class_dict={}
        count = 0
        term_index =""
        for word, tag in zip(words,tags):
            if re.search("^B",tag):
                count += 1
                entity_class = re.sub("B-","",tag)

                term_index = "T"+str(count)
                term_dict[term_index] = [word]
                entity_class_dict[term_index] = entity_class
                word_tagged.append(word+"__"+tag+"__"+term_index)   #mild__B-Participant__T29 
            elif re.search("^I",tag):
                word_tagged.append(word+"__"+tag+"__O")             # HFABP__I-Outcome__O
                term_dict[term_index].append(word)
            else:
                word_tagged.append(word)
        for key in term_dict.keys():
            term = " ".join(term_dict[key])
            term_dict[key] = term
        text_tagged = " ".join(word_tagged)
        if ("B-Count" not in tags) and ("B-Observation" not in tags):
            return [],[],term_dict,entity_class_dict
        relation_list = self._text2relation(sent_id, term_dict, text_tagged)
        for relation in relation_list:
            examples.append(InputExample(guid=relation[0], text_a=relation[1], text_b=None, label=relation[2]))
        else:
            return examples, relation_list,term_dict,entity_class_dict


class PICOProcessor(BlueBERTProcessor):
    def get_labels(self):
        """See base class."""
        return ["0","1"]

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    '''
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    '''
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature





def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        #tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn

class MED():
    def __init__(self):
        """ 
        Parse Medical Evidence Dependency MED
        """
        print ("Loading Medical Evidence Dependency model..." )
    
    def get_processor(self,task_name="pico"):

        processors = {"pico":PICOProcessor}
        tokenization.validate_case_matches_checkpoint(do_lower_case,config.init_checkpoint_dependency)
        bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

        if config.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (config.max_seq_length, bert_config.max_position_embeddings))


        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        processor = processors[task_name]()
        return processor

    def get_estimator(self,processor):
        tokenization.validate_case_matches_checkpoint(do_lower_case,config.init_checkpoint_dependency)
        bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)
        tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=do_lower_case)
        tpu_cluster_resolver = None
        
        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=None, #.master,
            model_dir=config.bluebert_dependency_dir,
            save_checkpoints_steps=1000, #FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000, #FLAGS.iterations_per_loop,
                num_shards=8, #FLAGS.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        train_examples = None
        num_train_steps = None
        num_warmup_steps = None
        label_list = processor.get_labels()
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=config.init_checkpoint_dependency,
            learning_rate=config.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=False,
            use_one_hot_embeddings=False)
 
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=config.pred_batch_size,
            eval_batch_size=config.pred_batch_size,
            predict_batch_size=config.pred_batch_size)
        return estimator

"""
MED = MED()
MED_processor = MED.get_processor("pico")
MED_estimator = MED.get_estimator(MED_processor)
tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=FLAGS.do_lower_case)


sents= [['Thirty-five', 'patients', 'returned', 'to', 'SCS', '.'], ['We', 'conclude', 'that', 'CMZ', 'is', 'effective', 'in', 'peripheral', 'neuropathic', 'pain', '.'], ['Morphine', 'obviously', 'requires', 'larger', 'individually', 'titrated', 'dosages', 'than', 'those', 'used', 'in', 'this', 'study', 'for', 'results', 'to', 'be', 'adequately', 'interpreted', '.']]
sent_tags = [['O', 'O', 'O', 'O', 'B-Outcome', 'O'], ['O', 'O', 'O', 'B-Intervention', 'O', 'B-modifier', 'O', 'B-Participant', 'I-Participant', 'I-Participant', 'O'], ['B-Intervention', 'O', 'O', 'B-modifier', 'B-Outcome', 'I-Outcome', 'I-Outcome', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

    
#------------Prediction-------------#
print ("start parsing Medical Evidence Dependency...")
sent_id = 0
import sys
for words,tags in zip(sents,sent_tags):
    sent_id += 1
    predict_examples, relation_list, term_dict = MED_processor.get_examples_from_pico(words,tags,str(sent_id))
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(sys.argv[1], "predict.tf_record")
    label_list=MED_processor.get_labels()
    MED_processor.file_based_convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer, predict_file)
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)
    result = MED_estimator.predict(input_fn=predict_input_fn)
    for (i, prediction) in enumerate(result):
        p = prediction["probabilities"] #[  9.99999881e-01   1.60701248e-07]    label = "1"
        if p[0] > p[1]:
            pred = "1"
        else:
            pred = "0"
        print(relation_list[i],"------",pred)
"""
