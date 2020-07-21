# coding: utf-8

import os, re,  string
import sys,codecs
from general_utils.umls_tagging import get_umls_tagging
import json

import warnings
warnings.filterwarnings("ignore")

def generate_XML(NERxml_dir,matcher,use_UMLS = 0,crfresult_dir="temp.conll"):
    crfresult_input = codecs.open(crfresult_dir,'r')
    NERxml_output=codecs.open(NERxml_dir,'w')
    if use_UMLS ==0:
        sents,entities_result=conll2txt_no_umls(crfresult_input)
    else:
        sents,entities_result=conll2txt(crfresult_input,matcher)
    entity_lists=['Participant','Intervention','Outcome']
    attribute_lists=['modifier','measure']
    NERxml_output.write("<?xml version=\"1.0\"?>")
    NERxml_output.write("<root>\n\t<abstract>\n")
    j=0

    for index,(sent, entities_forsent) in enumerate(zip(sents, entities_result)):
        if sent == "":
            continue
        if sent == "END":
            NERxml_output.write("\t</abstract>\n\n\t<abstract>\n")
            continue
        clean_sent=clean_txt(sent)

        pattern='class=\'(\w+)\''
        entities=entities_forsent.split('\n\t\t')
        new_entities=[]
        for e in entities:
            if e =='':
                new_entities.append('\n')
                continue
            match=re.search(pattern,e)
           
            if match.group(1) in attribute_lists:

                p1='\<entity'
                p2='entity\>'
                new=re.sub(p1,'<attribute',e)
                new=re.sub(p2,'attribute>',new)
                new_entities.append(new)
            else:

                new_entities.append(e)
        entities="\n\t\t\t".join(new_entities)
        entities=re.sub("\t\t\t\n$","",entities)
    
        NERxml_output.write("\t\t"+"<sent>\n"+"\t\t\t<text>"+clean_sent+"</text>\n")
        NERxml_output.write("\t\t\t"+entities)
        NERxml_output.write("\t\t"+"</sent>\n")
        j+=1
    NERxml_output.write("\t</abstract>\n</root>\n")
    rm_command = "rm "+crfresult_dir
    #os.system(rm_command)


def generate_json(out_text, out_preds,matcher,pmid="",sent_tags=[],entity_tags=["Participant","Intervention","Outcome"],attribute_tags=["measure","modifier","temporal"],relation_tags=[]):
    #abstract{ pmid; sent{section}; {entity{class;UMLS;negation;Index;start};relation{class;entity1;entity2}}
    results = {}
    results["pmid"] = pmid
    results["sentences"]={}
    #json_r=json.dumps(results)
        
    sent_id = 0
    entity_id=0
    attribute_id=0
    
    for sent, pred in zip(out_text, out_preds):
                
        sent_id+= 1
        sent_header = "sent_"+str(sent_id)
        results["sentences"][sent_header]={"Section":"METHODS","text":" ".join(sent),"entities":{},"relations":{}}
        
        indices_B = [i for i, x in enumerate(pred) if x.split("-")[0] == "B"]
        
        for ind in indices_B:  

            ''' retrieve all info for Enities and Attributes:
            "entity1":{                       
                       "text":"infliximab",
                       "class":"Intervention",
                       "negation":"0",
                        "UMLS":"",
                        "index":"T1",
                        "start":"19" 
            }
            '''
            entity_class = pred[ind].split("-")[1] # class
            if entity_class in entity_tags:        # header
                entity_id+=1
                entity_header = "entity_"+str(entity_id)
            else:
                attribute_id+=1
                entity_header = "attribute_"+str(attribute_id)
            start = ind
            inds=[]
            while(start < len(pred) and pred[start] !="O"):
                inds.append(start)
                start+=1
                if start in indices_B:
                    break
            
            c = [ sent[i] for i in inds]
            term_text = " ".join(c) # text
            #============= Negation =====================
            neg = 0
            
            #==============Negation END===================
            
            
            #============== UMLS encoding ================
            taggings = get_umls_tagging(term_text, matcher)
            umls_tag=""
            if taggings:
                for i,t in enumerate(taggings):
                    if i ==0:
                        umls_tag =str(t["cui"])+":"+str(t["term"])
                    else:
                        umls_tag = umls_tag +","+str(t["cui"])+":"+str(t["term"])
            #===============UMLS EDN =====================
            
            
            results["sentences"][sent_header]["entities"][entity_header]={"text":term_text,"class":entity_class,"negation":neg, "UMLS":umls_tag,"start":ind }

     
        #=============Relations ======================
        '''
         "relations":{
            "rel1":{
                "class":"has_value",
                "left":"T1",
                "right":"T2"
                }
            }
        }
        '''    
        #============END =============================
        
        
    json_r=json.dumps(results)
    return json_r

# sent_dict[sent_id] = sent_text
# term_dict[term_id] = entity
# entity_class_dict[term_id] = class
# relation_list[sent_id] = [relation_id, sent_w_tag, label]


def generate_json_from_sent(sent_id,sent_text, term_dict, entity_class_dict, relation_list=[], sent_tag="METHODS", umls_matcher=None): # formulate json object for one sentence
    results = {}

    results["sent_id"] = sent_id
    results["text"]= sent_text
    
    results["Evidence Elements"]={}
    results["Evidence Propositions"]={}
    observation_class=[]
    for term_id in term_dict.keys():
        pico_element = term_dict[term_id]
        pico_id = "s"+str(sent_id)+"_"+term_id
        pico_class = entity_class_dict[term_id]
        if pico_class not in ["Outcome", "Intervention","Participant"]:
            observation_class.append(term_id)
            continue
        #print (sent_text,"===",pico_element)
        start_index = sent_text.index(pico_element)
        umls_tag=""
        neg = False
        results["Evidence Elements"][pico_id]={"term":pico_element,"class":pico_class,"negation":neg, "UMLS":umls_tag,"start_index":start_index }
    
    mep= 0
    for observation_id in observation_class:
        #['11.T1.T5', '11.T1.T6', '11.T4.T6']
        mep +=1
        for i in relation_list:
            another = ""
            if observation_id == i.split(".")[1]:
                another = i.split(".")[2]
            elif observation_id == id.split(".")[2]:
                another = i.split(".")[1]
            else:
                continue

        mep_id= "s"+str(sent_id)+"_M"+str(mep)
        results["Evidence Propositions"][mep_id]={"Intervention":"","Outcome":"","Observation":""}
        mep +=1
    return results

    """ METHODS: We examined the association between hydroxychloroquine use and intubation or death at a large medical center in New York City"""
def aggregate(doc_id,abstract_text,sent_json,sentence_tags=[]):
    results = {}
    results["doc_id"] = doc_id
    results["abstract"] = abstract_text
    results["Evidence Map"] = {}
    results["Evidence Map"]["study design"]={"Participant":"","Intervention":"", "Outcome":"","Hypothesis":""}
    results["Evidence Map"]["study results"]=""
    results["sentences"]={}
    for sent_id in sent_json.keys():
        if len(sentence_tags)<1:
            sent_tag="METHODS"
        else:
            sent_tag = sentence_tags[sent_id]
        sent_ann = sent_json[sent_id]
        results["sentences"][sent_id]={"Section":sent_tag,"Evidence":sent_ann}

    
    json_r=json.dumps(results)
    return json_r
