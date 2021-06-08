# MD-informed Self-Attention :octocat: 
We introduce ***`Medical Evidence Dependency(MD)-informed Self-Attention`***, a Neuro-Symbolic Model for understanding free-text medical evidence in literature. We hypothesize this method can get the best of both: the high capacity of neural networks and the rigor, semantic clarity and reusability of symbolic logic.  

* Citation: Kang, T., Turfah, A. Kim, J. Perotte, A. and Weng, C. (2021). *[A Neuro-Symbolic Method for Understanding Free-text Medical Evidence.](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocab077/6270974?login=true)* Journal of the American Medical Informatics Association (in press)
* Contact: [Tian Kang](http://www.tiankangnlp.com)  (tk2624@cumc.columbia.edu)  
* Affiliation: Department of Biomedical Informatics, Columbia Univerisity ([Dr. Chunhua Weng](http://people.dbmi.columbia.edu/~chw7007/)'s lab)   

### Repository 
`MDAtt.py` generate Medical Evidence Dependency-information attention head.  
`MED_modeling.py`: modified from `bert/modeling.py` (attention_layer, transformer and bert classes).  
`run_MDAttBert.py`: run BERT with Medical Evidence Dependency-information attention head.  

### Model description
**Model** We develop a symbolic compositional representation called `Medical evidence Dependency` (MD) to represent the basic medical evidence entities and relations following the PICO framework widely adopted among clinicians for searching evidence. We use Transformer as the backbone and train one head in the Multi-Head Self-Attention to attend to *`MD`* and to pass linguistic and domain knowledge onto later layers (`MD-informed`). We integrated MD-informed Attention into BioBERT and evaluated it on two public MRC benchmarks for medical evidence from literature: i.e., [Evidence Inference 2.0](http://evidence-inference.ebm-nlp.com/) and [PubMedQA](https://pubmedqa.github.io/).   


**`Medical Evidence Dependency (MD) and Proposition`**   
   <img src="https://github.com/Tian312/MD-Attention/blob/master/figures/C6-MEP.png" alt="MEP" width="400"/>  

**`Medical Evidence Dependency (MD) Matrix`**   
  <img src="https://github.com/Tian312/MD-Attention/blob/master/figures/C6-MDmatrix.png" alt="MEP" width="300"/>  

**`Medical Evidence Dependency (MD)-informed Self Attention`**  
  <img src="https://github.com/Tian312/MD-Attention/blob/master/figures/C6-MDattention.png" alt="MEP" width="600"/> 


**Results** The integration of `MD-informed Attention` head improves BioBERT substantially for both benchmarks—as large as by +30% in the F1 score—and achieves the new state-of-the-art performance on the Evidence Inference 2.0. By visualizing the weights learned from MD-informed Attention head, we find the model can capture clinically meaningful relations separated by long passages of text. 
