"""
May, 2020
author = "Tian Kang, Columbia University"
email = "tk2624@cumc.columbia.edu"

1. Sentence classifiction: Title, Ojective, Background, Methods, Results, Conclusion
2. Evidence Element: extracting PICO elements from abstracts in clinical literature
3. Evidence Proposition: Medical Evidence Dependency parsing +formulate Medical Evidence Proposition based on PICO and MED
4. Evidence Map: Merge Evidence Propositions by study arms into Study Design and Study Results sections
"""

"""
This script is modified from run_parser.py to generate edge score matrix for Medical Evidence-informed Self-Attention
"""

import logging
import os, re, sys, codecs
import collections, pickle
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from general_utils import formalization,tf_metrics,format_predict

import numpy as np
np.set_printoptions(threshold=np.inf)
from bert import modeling
from bert import optimization
from bert import tokenization
from parser_config import Config


config=Config()
tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file, do_lower_case=False)
from nltk import sent_tokenize
flags = tf.flags
FLAGS = flags.FLAGS

from src.PICO_Recognition import PICO
from src.Medical_Evidence_Dependency import MED
PICO = PICO()
PICO_processor = PICO.get_processor()
PICO_estimator = PICO.get_estimator(PICO_processor)
MED = MED()
MED_processor = MED.get_processor()
MED_estimator = MED.get_estimator(MED_processor)


import numpy as np
from random import random
alpha = 1 # the probability to save a pair of dependency relations for MDAtt-BERT


class MDAtt():
    def __init__(self):
        """ create an attention head for Transformer-based model by Medical Evidence Proposition """
    def tensor2tokens(input_ids):
        1
    def dependency_matrix(self, words, relation_list_postive, term_dict, entity_class_dict ):
        
        """
        Example input:
        1 ONCLUSIONS : In this observational study involving patients with Covid-19 who had been admitted to the hospital , hydroxychloroquine administration was not associated with either a greatly lowered or an increased risk of the composite end point of intubation or death .
        ['1.T3.T4', '1.T4.T5']
        {'T1': 'Covid-19', 'T2': 'admitted to the hospital', 'T3': 'hydroxychloroquine administration', 'T4': 'greatly lowered or an increased', 'T5': 'risk of the composite end point of intubation or death'}
        {'T1': 'Participant', 'T2': 'Participant', 'T3': 'Intervention', 'T4': 'Observation', 'T5': 'Outcome'}

        direction: Outcome -> Observation/Count -> Intervention
        root Intervention is dependent on itself
        """ 
        import itertools
        def _get_loc(term_index, words, term_dict):
            """
            loc: the index of the first word of the term in the sent words
            """
            next_index = "T"+str(int(re.sub("T","",term_index))+1)
            if next_index in term_dict.keys():
                next_loc = " ".join(words).rindex(" "+term_dict[next_index])
                phrase = " ".join(words)[:next_loc]
            else:
                phrase = " ".join(words)
            loc_in_sent = phrase.rindex(term_dict[term_index])
            if loc_in_sent >0:
                loc = len(re.split("\s+",phrase[:(loc_in_sent)]))-1
            else:
                loc = 0 

            return loc

        dim = len(words)
        edge_score_matrix = np.zeros((dim,dim)) # (term1, term2) 
        
        # add 1 to all intervention
        for idx in entity_class_dict.keys():
            if entity_class_dict[idx] == "Intervention":
                loc_1 = _get_loc(idx, words, term_dict)
                loc_list_1 = list(range(loc_1,len(term_dict[idx].split(" "))+loc_1))
                coord_intervention = itertools.product(loc_list_1, loc_list_1)
                for c in coord_intervention:
                    edge_score_matrix[c[0],c[1]] = 1


        hierarchy={"Intervention":3,"Observation":2,"Count":1,"Outcome":0}
        for pair in relation_list_postive:
            
            if hierarchy[entity_class_dict[pair.split(".")[1]]]>hierarchy[entity_class_dict[pair.split(".")[2]]]:
                index_1 = pair.split(".")[1]
                index_2 = pair.split(".")[2]
            else:
                index_1 = pair.split(".")[2]
                index_2 = pair.split(".")[1]

            loc_1 = _get_loc(index_1, words, term_dict)
            loc_2 = _get_loc(index_2, words, term_dict)
            loc_list_1 = list(range(loc_1,len(term_dict[index_1].split(" "))+loc_1))
            loc_list_2 = list(range(loc_2,len(term_dict[index_2].split(" "))+loc_2))

            """ term 1: parent term loc_list_1 e.g. intervention
                term 2: child term  loc_list_2 e.g. observation/count
                1. assign 1 to edge_score_matrix[loc_1, loc_2] = 1
                2. if term 1 is intervention: 
                    edge_score_matrix[loc_1,loc_1] = 1
                
            """
            coord = itertools.product(loc_list_1, loc_list_2)
            for c in coord:
                #print (c,words[c[0]],words[c[1]],words[c[0]]+"-"+words[c[1]] )
                edge_score_matrix[c[0],c[1]] = 1 #str(words[c[0]]+"-"+words[c[1]])

        return edge_score_matrix


    def get_matrix_from_one_sent(self, sent_text,
                                 PICO_processor,PICO_estimator,MED_processor,MED_estimator):
        temp_out= os.path.join(os.getcwd(),"temp_out") 
        if not os.path.exists(temp_out):
            try:
                createdir= "mkdir "+temp_out
                os.system(createdir)
            except:
                print("DIR ERROR! Unable to create this directory!")
                raise

        #start parsing
        with tf.gfile.Open(os.path.join(config.bluebert_pico_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            pico_label_list = PICO_processor.get_labels()
            #print ("sent_text:",sent_text)
            predict_examples = PICO_processor.get_pred_examples(sent_text, raw_text = True)
            predict_file = os.path.join(temp_out, "PICO.predict.tf_record")
            PICO_processor.filed_based_convert_examples_to_features(
                    predict_examples, 
                    pico_label_list,  
                    max_seq_length = config.max_seq_length, 
                    tokenizer = tokenizer, 
                    output_file = predict_file,
                    output_dir = os.getcwd(),
                    mode = "test")
            predict_input_fn = PICO_processor.file_based_input_fn_builder(
                    input_file=predict_file,
                    seq_length=config.max_seq_length,
                    is_training=False,
                    drop_remainder=False)
            PICO_result = list(PICO_estimator.predict(input_fn=predict_input_fn))
            sents, sent_preds = PICO_processor.result_to_pair_for_return(predict_examples, PICO_result, id2label)
            #print (sents,sent_preds)
            label_list=MED_processor.get_labels()
            
            sent_len = len(sent_text)
            sent_matrix = np.zeros((sent_len, sent_len))
            sent_matrix_dropout = np.zeros((sent_len, sent_len))
            new_sents = []
            cur_length = 0
            for words, tags in zip(sents, sent_preds):    
                tags = format_predict.check_IOB(tags)
                predict_examples, relation_list, term_dict, entity_class_dict = MED_processor.get_examples_from_pico(words,tags,"sent_0")
                predict_file = os.path.join(temp_out, "MED.predict.tf_record")
                MED_processor.file_based_convert_examples_to_features(predict_examples, label_list,64, tokenizer, predict_file)
                predict_input_fn = MED_processor.file_based_input_fn_builder( #share
                        input_file=predict_file,
                        seq_length=64,
                        is_training=False,
                        drop_remainder=False)
                MED_result = MED_estimator.predict(input_fn=predict_input_fn)
                relation_list_postive=[]
                for (i, prediction) in enumerate(MED_result):
                    p = prediction["probabilities"]
                    pred = "1" if p[0] < p[1] else "0"
                    if pred == "1":
                        relation_list[i][-1] = pred
                        relation_list_postive.append(relation_list[i][0])
                print ("\n==== parsing results for one sent =====")
                print ("sent:"," ".join(words))
                print ("PICO entities: ", term_dict,entity_class_dict)
                print ("Evidence Dependency: ",relation_list_postive,"\n")
                
                # set probability to drop a depdendecy relation
                #relation_list_postive_dropout = [ relation for relation in relation_list_postive if alpha > random()]
                
                
                matrix = self.dependency_matrix(words, relation_list_postive, term_dict, entity_class_dict)
                #matrix_dropout = self.dependency_matrix(words, relation_list_postive_dropout, term_dict, entity_class_dict)
                r = cur_length 
                c = cur_length 
                sent_matrix[r:r+matrix.shape[0], c:c+matrix.shape[1]] += matrix
                #sent_matrix_dropout[r:r+matrix_dropout.shape[0], c:c+matrix_dropout.shape[1]] += matrix_dropout
                cur_length = cur_length + matrix.shape[0]
                new_sents.append(" ".join(words))
            return  new_sents, sent_matrix

    def onesentann2matrix(self, sent_ann):
        one_sent = re.sub("##","",sent_ann)
        sents, one_matrix = self.get_matrix_from_one_sent(one_sent,PICO_processor,PICO_estimator,MED_processor,MED_estimator)

        #assert len(" ".join(sents).split(" ")) == len(sent_ann.split(" "))

        # first expand rows according to wordpieces and then columns
        new_matrix = []
        for idx, w in enumerate(sent_ann.split(" ")):
            wp = w.split("##")
            if len(wp) <= 1:
                if idx == 0:
                    new_matrix = one_matrix[0,:]
                else:
                    new_matrix = np.vstack((new_matrix, one_matrix[idx,:]))
            else:
                if idx == 0:
                    new_matrix = np.array([one_matrix[0,:],]*len(wp))
                else:
                    temp = np.array([one_matrix[idx,:],]*len(wp))
                    new_matrix = np.vstack((new_matrix,temp))
        final_matrix = []
        for idx, w in enumerate(sent_ann.split(" ")):
            wp = w.split("##")
            if len(wp) <= 1:
                if idx == 0:
                    final_matrix = new_matrix[:,0].reshape([new_matrix.shape[0],1])
                else:
                    final_matrix = np.hstack((final_matrix, new_matrix[:,idx].reshape([new_matrix.shape[0],1])))
            else:
                if idx == 0:
                    final_matrix = np.array([new_matrix[:,0],]*len(wp)).transpose()
                else:
                    temp = np.array([new_matrix[:,idx],]*len(wp)).transpose()
                    final_matrix = np.hstack((final_matrix,temp))
        return final_matrix

    def question2matrix(self, q_ann):
        
        # ['[', 'o', ']', 'peak', 'ins##pi##rator##y', 'pressure', '(', 'p##ip', ')', '[', 'i', ']', 'c##isa##p##ride', '(', '10', 'mg', ',', 'via', 'na##so##gas##tric##ube', ')', '[', 'c', ']', 'place##bo']
        
        
        q_ann_list = re.split("\s+", q_ann)
        q_matrix = np.zeros((len(q_ann_list),len(q_ann_list)))
        #print (q_matrix.shape)
        def find_sub_list(sl,l):
            results=[]
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    results.append((ind,ind+sll-1))
        
            return results

        (i_start,i_end) = find_sub_list(["[","i","]"],q_ann_list)[0]
        (c_start,c_end) = find_sub_list(["[","c","]"],q_ann_list)[0]
        #print (len(q_ann_list),i_start,i_end, c_start,c_end)
        q_matrix[(i_end+1):c_start,(i_end+1):c_start]=1
        q_matrix[(c_end+1):,(c_end+1):] =1
        #print (" * Question_tokens:",q_ann)
        #print (" * Question matrix: ", q_matrix)
        return q_matrix

    def example2matrix(self, example_toks, guid, ab_matrix_dict,padding = True,max_seq_length=512):
        """
        tokens= ['[CLS]', 'Is', 'the', 'protein', 'Pa', '##pi', '##lin', 'secret', '##ed', '?', '[SEP]', 'Pa', '##pi', '##lin', ':', 'a', 'Dr', '##oso', '##phi', '##la', 'pro', '##te', '##og', '##ly', '##can', '-', 'like', 'su', '##lf', '##ated', 'g', '##ly', '##co', '##p', '##rote', '##in', 'f', '##r', 'o', '##m', 'basement', 'membrane', '##s', '.', 'A', 'su', '##lf', '##ated', 'g', '##ly', '##co', '##p', '##rote', '##in', 'was', 'isolated', 'from', 'the', 'culture', 'media', 'of', 'Dr', '##oso', '##phi', '##la', 'K', '##c', 'cells', 'and', 'named', 'p', '##ap', '##ili', '##n', '.', 'A', '##ffin', '##ity', 'pu', '##rified', 'antibodies', 'against', 'this', 'protein', 'localized', 'it', 'p', '##rim', 'a', '##rily', 'to', 'the', 'basement', 'membrane', '##s', 'of', 'em', '##b', '##ryo', '##s', '.', '[SEP]'] 
        """
        max_seq_matrix = np.zeros([max_seq_length, max_seq_length])
        example_toks.pop(0) # [CLS]
        example_toks.pop(-1) # [SEP]
        text = " ".join(example_toks)
        new_text = re.sub("\s+##","##",text)
        [q_ann,ab_ann] = new_text.split(" [SEP] ")
        
             
        print ("q_ann:",q_ann)
        # get MED matrix for question
        if re.search("^\[ o \]",q_ann): # evidence inference data
            q_matrix = self.question2matrix(q_ann)
        else:
            q_matrix = self.onesentann2matrix(q_ann)
        r = 1
        c = 1
        
        max_seq_matrix[r:r+q_matrix.shape[0], c:c+q_matrix.shape[1]] += q_matrix
        
        print ("q_matrix:", q_matrix.shape, "c",c, "r:",r )
        # get MED matrix for abstract first search if has been calculated
        guid = guid.split("-")[1]
        if guid in ab_matrix_dict.keys():
            
            print ("SEEN id:", guid )
            ab_matrix = ab_matrix_dict[guid]
        else:
            print ("UNSEEN id:", guid ) 
            
            ab_matrix = self.onesentann2matrix(ab_ann)
            ab_matrix_dict[guid] = ab_matrix
            
        try: 
            c = c+ 1+ q_matrix.shape[0]
            r = r+ 1+ q_matrix.shape[0]
            print ("ab_matrix:", ab_matrix.shape, "c",c, "r:",r )
            max_seq_matrix[r:r+ab_matrix.shape[0], c:c+ab_matrix.shape[1]] += ab_matrix
        except:
            if guid in ab_matrix_dict.keys():
                print ("Seen id, but failed in reuse...")
                ab_matrix = self.onesentann2matrix(ab_ann)
                ab_matrix_dict[guid] = ab_matrix
                c = 1+1+ q_matrix.shape[0]
                r = 1+ 1+ q_matrix.shape[0]
                max_seq_matrix[r:r+ab_matrix.shape[0], c:c+ab_matrix.shape[1]] += ab_matrix
            else :
                print ("!! unble to process",guid,"!!!\n")

        return max_seq_matrix, ab_matrix_dict

    
    
"""
tokens = ['[CLS]', '[', 'o', ']', 'scores', 'indicating', 'under', '##sed', '##ation', '[', 'i', ']', 'intermittent', 'di', '##az', '##ep', '##am', '[', 'c', ']', 'continuous', 'mid', '##az', '##ola', '##m', '[SEP]', 'scores', 'indicating', 'under', '##sed', '##ation', 'were', 'more', 'common', 'with', 'di', '##az', '##ep', '##am', '(', 'p', '<', '00', '##1', ')', ';', 'overall', 'adequate', 'se', '##dation', 'mid', '##az', '##ola', '##m', '64', '##7', '%', ',', 'di', '##az', '##ep', '##am', '35', '##7', '%', '(', 'p', '=', '02', '##1', ')', '.', 'no', 'patient', 'exhibited', 'inappropriate', '##ly', 'prolonged', 'se', '##dation', '.', 'cost', 'was', ':', 'mid', '##az', '##ola', '##m', 'au', '##s', '$', '198', '/', 'h', ';', 'di', '##az', '##ep', '##am', 'au', '##s', '$', '00', '##6', '/', 'h', '.', 'conclusion', ':', 'both', 'regime', '##ns', 'produced', 'rapid', 'onset', 'of', 'acceptable', 'se', '##dation', 'but', 'under', '##sed', '##ation', 'appeared', 'more', 'common', 'with', 'the', 'cheaper', 'di', '##az', '##ep', '##am', 'regime', '##n', '.', 'at', 'least', '140', 'patients', 'should', 'be', 'studied', 'to', 'provide', 'evidence', 'applicable', 'to', 'the', 'general', 'i', '##cu', 'population', '.', 'used', 'alone', ',', 'a', 'se', '##dation', 'score', 'may', 'be', 'an', 'inappropriate', 'outcome', 'measure', 'for', 'a', 'se', '##dation', 'trial', '.', '[SEP]'] 

import time
MDAtt=MDAtt()
before = time.time()
matrix = MDAtt.example2matrix(tokens,max_seq_length=128)
cost = time.time()-before
print (matrix)
print(cost)
"""

