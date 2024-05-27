# Social Complexity Lab

We proceed in two steps:
1. we trained a gradient boosting algorithm (xgboost):
 a. to have a strong baseline 
 b. to evalute the predictive power of each questions
2. our main model is an autoencoder-tabular blabla:
a.
b.

## Gradient Boosting as a Baseline

We use xgboost, the most important parameters for the prediction task are,
1. ```scale_pos_weight``` to account for the unbalanced outcome
2. ```reg_lamba``` for L2 regularization to avoid overfitting

xgboost need only the top X questions to achieve agood prediction and saturater when we add more features

<img src="figures/feature_importance.png" alt="drawing" width="700"/>

<img src="figures/f1_score_distribution.png" alt="drawing" width="700"/>

## TabularEncoder




## Data Processing
### Embedding and tokenisation
In order for the deep learning models to understand the data, we must embed it. This requires us to put an integer value to all questions and answers. 

For questions, it is easy: We just group questions asked in multiple surveys and assign each group an integer. 

For the answers, we do this through **tokenisation**, where we convert each answer to a `token`.   
For **categorical** (`data_processing/categorical.py`) answers, we assign each unique answer (accounting for the rephrasing) an integer. This creates the same mapping for an answer (e.g. "yes") that answers two different questions. This allows us to share the vocabulary, decreasing model size and increasing the semantic meaning of each answer.   
For **numeric** or **date or time** (`data_processing/numeric_and_date.py`) we bin the values into percentiles (100 bins), that is again shared between questions to increase the semantic meaning. 

If an individual has not answered a given question, we assign their answer to an unknown token `[UNK]`. 

### Sequential input
As our models require sequential data (so they better fit registry data in phase 2), we must convert the tabular format into sequences. We create a sequence for each year to reflect the temporal aspect of the data. The order of the sequence for each year is kept constant across years, as we have no predefined order. This is done using the `codebook`, where we allocate each question (e.g. cfxxx003) a specific spot in the sequence. This means the question "Gender respondent" will always be at the same index of the sequence across all years. For registry data, the sequence would be ordered in the order in which they appear throughout an individual's life.  

If a question has not been asked for a given year, we still keep its spot in the sequence and assign an unknown token `[UNK]` as the answer for all individuals. 

For the ExcelFormer, we order the sequence based on the feature importance from the XGBoost.

We also allow for subsetting the number of columns through the `custom_pairs` in `data_processing/pipeline.py`. 





