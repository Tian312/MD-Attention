from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

""" Modified BERT with MDAtt: Medicalevidence Dependency-informed self-Attention """


import collections
import csv,re
import logging
import os,codecs
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES']='3,2,1'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3,0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import MED_modeling as modeling
#from bert import modeling
from bert import optimization
from bert import tokenization
from parser_config import Config
config=Config()

from random import random
# Medical Evidence_informed Self-attention:


flags = tf.flags
FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_float(
        "mask_rate", 1,
        "MD-attention input_mask rate. 0 for no attending at all; 1 for no masking. "
        "for the task.")
flags.DEFINE_string(
        "data_dir", None,
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                                        "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_bool("do_prepare", False, "Whether to run file-based data to feature process ")
flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
        "do_predict", False,
        "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_med_informed", False, "Whether to add med_informed self-attention head")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                                     "Total number of training epochs to perform.")

flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                                         "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                                         "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

tf.flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample. combination of run_classifier and bluebert

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 med_matrix,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.med_matrix = med_matrix

        
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

class QAProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version). tsv file"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "-1"]
    
    def preprocess_values(self,text):   # remove all the decimals
        nums=re.findall('\d\.\d',text)

        new_text = text
        for num in nums:
            gang=re.search('(\d+)\.(\d+)',num)
            new_text=re.sub(gang.group(1)+'.'+gang.group(2),gang.group(1)+gang.group(2),new_text)
        
        new_text = re.sub("\s+vs\s*\.\s+"," vs ", new_text)
        return new_text
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            
            text_a = tokenization.convert_to_unicode(self.preprocess_values(line[1]))
            text_b = tokenization.convert_to_unicode(self.preprocess_values(line[2]))
            
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

""" TO DO: modify SQUAD processor """
class SQUADProcessor(DataProcessor):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")),
                "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    
    def read_squad_examples(input_file, is_training):
        """Read a SQuAD json file into a list of SquadExample."""
        with tf.gfile.Open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:

                        if FLAGS.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                    "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the # document. If this CAN'T happen it's likely due to weird Unicode# stuff so we will just skip the example.#
                            # Note that this means for training mode, every example is NOT # guaranteed to be preserved.
                            actual_text = " ".join(
                                    doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                    tokenization.whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            is_impossible=is_impossible)
                    examples.append(example)

        return examples

    

from MDAtt import MDAtt
tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
MDAtt = MDAtt()#max_seq_length = config.max_seq_length, tokenizer = tokenizer)
assert FLAGS.max_seq_length == config.max_seq_length

def MED_convert_single_example(example, label_list, max_seq_length,
                           tokenizer, ab_matrix_dict):
    """ convert a single "InputExample" into a single "InputFeature" """
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            med_matrix=np.zeros((max_seq_length,max_seq_length)),
            is_real_example=False)
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    guid = example.guid
    
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
    #   embedding vector (and position vector). This is not *strictly* necessary
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
    
    # ****** MASKING ********* 
    input_mask = [1] * len(input_ids)
    # here only mask MD-Attention head
    mask_rate = FLAGS.mask_rate # 1 for not masking 
    input_mask = [1 if mask_rate > random() else 0 for m in input_mask ]
    #print("input_mask:", input_mask)
    #***************************
    """ create Medical Evidence Dependency Matrix """
    
    
    med_matrix, ab_matrix_dict= MDAtt.example2matrix(tokens,guid, ab_matrix_dict,padding = True,max_seq_length=max_seq_length)
    
            
            
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert med_matrix.shape[1] == max_seq_length
    #print ("example:",example )    
    
    label_id = label_map[example.label]
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        med_matrix = med_matrix,
        is_real_example=True)
    
    print (" \n*** finished", guid,"....")
    return feature,ab_matrix_dict

def MED_file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, ab_matrix_dict):
    """Convert a set of `InputExample`s to a TFRecord file."""
    
    #  a dictionary to store matrix_for parsed document ab_matrix_dict
    
    
    
    writer = tf.python_io.TFRecordWriter(output_file)
    #error_log = codecs.open(output_file+".exceptions","w")
    
    import time
    before = time.time()
    print ("\n...",len(examples),"in total!")
    for (ex_index, example) in enumerate(examples):
        
        print (" \n*** processing", (ex_index+1),"th example:",example.guid,"....")
        current = time.time()
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
         
        if True:   
            feature, ab_matrix_dict =MED_convert_single_example(example, label_list,
                                         max_seq_length, tokenizer, ab_matrix_dict)
       
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature([feature.label_id])

            # reshape MED matrix into list 
            # [max_seq_length , max_seq_length]
            # [ max_seq_length * max_seq_length ,]
            features["med_matrix_tensor"] = create_int_feature(feature.med_matrix.reshape((max_seq_length*max_seq_length,)).astype('int'))
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            
            one_cost = time.time()-current
            current = time.time()
        #except:
        #    error_log.write(str(example.guid)+"\n")
    cost = time.time()-before
    
    writer.close()
    return ab_matrix_dict
        

def MED_file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        #"unique_ids":tf.FixedLenFeature([seq_length],tf.int64
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "med_matrix_tensor": tf.FixedLenFeature([seq_length*seq_length], tf.int64),
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

            
def create_model(bert_config, is_training, input_ids, med_matrix_tensor, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings, question_type = "multiple_choice"):
    """Creates a classification model for multiple choice question ."""
    model = modeling.MDAttBertModel(  #BertModel
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        med_matrix_tensor=med_matrix_tensor,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    
    
    """    get token-level model.get_sequence_output() 
    final_hidden = model.get_sequence_output()
    print ("create_model-final_hidden:",final_hidden)
    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
    print ("batch_size:",batch_size,"seq_length:",seq_length,"hidden_size:",hidden_size)
    #final_hidden_matrix = tf.reshape(final_hidden,
    #                               [batch_size * seq_length, hidden_size])
    
    final_hidden_matrix = final_hidden
    print ("final_hidden_matrix",final_hidden_matrix,"\n")
    """
    
    # get sequence-leverl 
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


def MED_model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, question_type="multiple_choice"):
    """Returns `model_fn` closure for TPUEstimator."""
    
    def model_fn(features, labels, mode, params):
        tf.logging.info("\n*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))
        #unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        med_matrix_tensor=features["med_matrix_tensor"]  # [batch_size, seq_length*seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]        
        label_ids = features["label_ids"]
        is_real_example = None
        
        print ("feature: label_ids",label_ids)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"],dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
            
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config = bert_config, 
            is_training = is_training, 
            input_ids = input_ids,
            med_matrix_tensor = med_matrix_tensor,
            input_mask = input_mask, 
            segment_ids = segment_ids, 
            labels = label_ids,
            num_labels = num_labels,
            use_one_hot_embeddings = use_one_hot_embeddings)
    
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names)= modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

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
        
        
        
        

    
# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    return input_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    return features


def main(_):
    
    
    
    tf.logging.set_verbosity(tf.logging.ERROR)
    processors = {
      "squad": SQUADProcessor,
      "tsv": QAProcessor
    }
    
    
    # ab_matrix_dic to store parsed results for the sake of processing time
    
    
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.") 
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    ab_matrix_dict={}

    
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    print("LLLLLLLlabel_list:", label_list)
    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
        
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=FLAGS.master,
          model_dir=FLAGS.output_dir,
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=FLAGS.iterations_per_loop,
              num_shards=FLAGS.num_tpu_cores,
              per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    
    
    
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    model_fn = MED_model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        question_type="multiple_choice"
    )
    
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if FLAGS.do_prepare:
            ab_matrix_dict = MED_file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, ab_matrix_dict)
            print ( "after train ab_matrix_dict: ", len(ab_matrix_dict))
        
        else:
            tf.logging.info("\n***** Running training *****")
        
            train_input_fn = MED_file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
    
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        
    
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())
    
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        if FLAGS.do_prepare:
            ab_matrix_dict= MED_file_based_convert_examples_to_features(
           eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file,ab_matrix_dict)
            print ( "after do Eval ab_matrix_dict: ", len(ab_matrix_dict))   
        else:
            print ("\n***** Running evaluation*****")   
        
            tf.logging.info("\n***** Running evaluation*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
            tf.logging.info("  Batch size = %d\n", FLAGS.predict_batch_size)
            # This tells the estimator to run through the entire set.
            eval_steps = None
            if FLAGS.use_tpu:
                assert len(eval_examples) % FLAGS.eval_batch_size == 0
                eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
        
            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = MED_file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)

            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)   

            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        
        
        if FLAGS.do_prepare:
            ab_matrix_dict = MED_file_based_convert_examples_to_features(predict_examples, label_list,
                                        FLAGS.max_seq_length, tokenizer, predict_file, ab_matrix_dict)
            tf.logging.info( "after do_predict ab_matrix_dict: ", len(ab_matrix_dict))
        
        else:
        
            tf.logging.info("\n***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                       len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)


            tf.logging.info("  Batch size = %d\n", FLAGS.predict_batch_size)
            predict_drop_remainder = True if FLAGS.use_tpu else False
            predict_input_fn = MED_file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)

            result = estimator.predict(input_fn=predict_input_fn)
            output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
            with tf.gfile.GFile(output_predict_file, "w") as writer:

                num_written_lines = 0
                tf.logging.info("\n***** Predict results *****")
                for (i, prediction) in enumerate(result):

                    probabilities = prediction["probabilities"]
                    print (i,probabilities)
                    #if i >= num_actual_predict_examples:
                    #    break
                    pos = np.argmax(probabilities)
                    if pos == 0:
                        pred = '0'
                    elif pos == 1:
                        pred = '1'
                    else:
                        pred = "-1"
                    output_line = "\t".join(str(class_probability) 
                        for class_probability in probabilities) + "\t"+pred+"\n"

                    writer.write(output_line)
                    num_written_lines += 1
            #assert num_written_lines == num_actual_predict_examples    

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
