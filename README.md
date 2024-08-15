# ASSIGNMENT-3
Assignment Report on Biassed Large Language Models (LLMs) 
By: Aswin Manoharan Kuniyil-22032995 
1.INTRODUCTION
The BERT-based uncased model is a variant of the BERT 
(Bidirectional Encoder Representations from  
Transformers) architecture, designed for natural language 
understanding tasks. In this model, all text is lowercased 
before processing, meaning it doesn't differentiate between 
uppercase and lowercase letters. This helps in generalising 
across different contexts where case sensitivity isn't 
crucial. The "uncased" model is particularly effective for 
tasks where the distinction between uppercase and 
lowercase letters is minimal, such as sentiment analysis or 
text classification, while maintaining robust performance 
in capturing the nuances of language (Devlin et al., 2019).  
2  Experimenting With the Bert Transformer Mode  
2.1  Dataset Overview 
This study uses _Spam_Multilingual_Collection dataset1 
collected from Hugging Face. It includes 5574 SMS only 
in English, labelled as spam or ham. However, it has 
evolved to cover multiple languages, which makes it 
resourceful for research in the area of spam in multilingual 
environments. This data is particularly suitable for an NLP 
task. 
2.2  Data Preprocessing  
Data preprocessing entails nothing more than converting 
raw data into a more comprehensible format (Pratap Singh 
and Kaur, 2022). Before analysis, as usual, text data 
undergo cleaning and normalization in order to achieve 
higher data quality.  
• Lowercasing: Translates all text to lowercase in 
order that there is uniformity and no contrast in the 
style of writing.  
• HTML Tag Removal: BeautifulSoup removes  
HTML tags, leaving only textual content.  
• Special Symbols and Numbers Removal: Uses a 
regular expression to remove special symbols and 
numerals while leaving just alphabetic letters and 
whitespace.  
• Label conversion: After text processing the labels 
are converted to integers  
2.3 Exploratory Data Analysis (EDA)  EDA provides 
important information about the dataset, including 
information about hidden patterns, relationships and 
outliers. (Saxena, 2023) Here’s an overview of the EDA 
performed for the SMS Spam Multilingual Collection 
dataset: In the figure 1 as a donut graph shows “spam” and “ham 
using color pink show the distribution of “ham” is 
relatively more dominant with a percentage of 86. 6 % as 
compared to “spam” of 13. 4%, respectively.  
Figure 2 shows two-word clouds: The first is for “spam” 
messages and the second for “ham” messages with the first 
the most prevalent words highlighted included ‘call,’ ‘free,’ 
and ‘text’ In contrast, the ‘ham’ message highlighting the 
following words ‘U,’ ‘i’m,’ and ‘go’.  
Figure 3 shows the Top 20 Bigrams in Text Data Bigrams 
are pairs of consecutive words. The chart lists various 
bigrams on the y-axis and their frequencies on the x-axis, 
ranging from 0 to 175. Each bar is colour-coded for clarity.  
Some of the most frequent bigrams include “of the,” “in 
the,” “to be,” and “I am.” This visualization helps in 
understanding common word pairs in a text dataset, useful 
for linguistic analysis or natural language processing tasks. 
2.4  Data splitting  
After preprocessing, the data is divided into training and 
validation, with 80% of the training and 20% of the 
validation.  
3  FINE-Tuning Masked Language Model   
3.1  Model Implementation  
The model implementation involves converting the SMS 
dataset into sequences and classification with BERT. 
(Mozafari, Farahbakhsh and Crespi, 2020)   
• Tokenization: The BERT tokenizer (Bert  
Tokenizer) is loaded to convert text data into 
token IDs, applying padding and truncation to 
ensure uniform input lengths. The data is then 
transformed from pandas Data Frames into 
Hugging Face Dataset objects, tokenized, and 
formatted for PyTorch.  
• Model  Loading:  The  
BertForSequenceClassification model is loaded 
from the database2  Hate speech detection and 
racial bias mitigation insocial media basedon  
BERT modelbert” repository, a variant of BERT 
fine-tuned for binary classification tasks   
• Training Configuration: Training arguments are 
defined, including the batch size(64), number of 
epochs(5), and logging steps(10). These arguments 
are given with the Trainer, together  
with the model, tokenized datasets, and the 
functions in charge of the metrics computation.  
• Training Execution: The given model is trained 
with the help of the Trainer class. The model is 
assigned to classify the given SMS messages into 
spam and legitimate messages. During training, a 
variety of skills, such as accuracy and F1 scores, 
have to be used for model assessment.  
3.2  Model Evaluation  
The model evaluation examines its performance using a 
variety of Metrics and visualizations. 
Figure 4 presents a graph of the Training and Validation 
Loss plot over time, with two lines: a blue line representing 
"Training Loss" and a red line representing "Validation 
Loss." The x-axis covers discrete intervals (steps) during 
the training process, ranging between 0 and 35, while a 
yaxis shows the loss value, which ranges from 0 to 0.12. 
The blue training loss line starts just below 0.5, decreases 
sharply, and then oscillates up and down throughout the 
training process. Meanwhile, the red validation loss line 
begins just above 0.10, spikes to around 0.11, and generally 
trends downward with some fluctuations.  
3.3 Model performance o Accuracy:  
0.9901345291479821 o Precision:  
0.9662162162162162 o Recall:  
0.959731543624161 o F1 Score:  
0.962962962962963 o ROC AUC:  
0.9943863159503662  
The model's performance metrics are as follows: accuracy 
is 99%, Precision is 96.6%, and Recall is 95.9%. The F1 
Score is 96.2%, and the ROC AUC is 99.4%.  
3.4  Visualizations  
Figure 5: Confusion matrix of BERT-based uncased 
model  
Figure 6: ROC curve for Bert-based uncased model  
Figure 5 shows the confusion matrix, in which two classes,  
‘0’ and ‘1’, represent the model. It displays 961 false 
negatives, 5 true negatives, 6 true positives, and 143 false 
positives.   
A graph depicting the model's ROC curve is shown in 
Figure 6. On one side, we have the TPR, which likewise 
ranges from 0.0 to 1.0, and on the other, we have the FPR, 
which shows a range from 0.0 to 1.0. There are two lines 
on the graph: a dashed diagonal line labeled “No Skill,” 
indicating random classification, and a solid red line 
labeled “ROC curve (area = 0.99),” suggesting excellent 
model performance with an AUC very close to 1. This 
shows high accuracy in distinguishing among classes  
References   
Devlin, J. et al. (2019) ‘BERT: Pre-training of deep 
bidirectional transformers for language understanding’, in 
NAACL HLT 2019 - 2019 Conference of the North 
American Chapter of the Association for Computational 
Linguistics: Human Language Technologies - Proceedings 
of the Conference. 
Mozafari, M., Farahbakhsh, R. and Crespi, N. (2020) ‘Hate 
speech detection and racial bias mitigation in social media 
based on BERT model’, PLOS ONE, 15(8), pp. 1–26. 
Available  at: 
https://doi.org/10.1371/journal.pone.0237861.  
Pratap Singh, A. and Kaur, N. (2022) ‘Introduction To Data 
Preprocessing: A Review’. Available at: 
https://doi.org/10.36227/techrxiv.21068668.v1.  
Saxena, B. (2023) ‘Exploratory Data Analysis And 
Visualization : Netflix Data Using Python Libraries’, 
11(10), pp. 296–301. 
  
