# Social Complexity Lab, PreFer Challenge [1]

We proceed in three steps:
1. **Gradient boosting algorithm (xgboost):**
    - To establish a strong baseline.
    - To evaluate the predictive power of each question.
2. **Our best performing model is the ExcelFormer [4] + RNN:**
    - (*ExcelFormer*) To create deep representation of the (answered) questionnaires
    - (*Recurrent Neural Network*) To leverage temporal interactions between the questionnaires
3. **Our main work has been on an AutoEncoder model:**
    - To capture complex, non-linear relationships in the data.
    - To explore new factors and interactions that traditional gradient boosting models might miss.

We have decided these steps for a few reasons. First, it is important to establish a solid baseline using traditional methods such as XGBoost. However, XGBoost doesn't scale as well with data, making it an unlikely candidate for phase 2 experiments. This is why we have worked with embedding strategies and AutoEncoders. These approaches allow us to model the data more similar to text (i.e. sequential data), enabling pretraining like seen in fields as NLP and CV. These fields have been revolutionized by self-supervised learning (SSL) and big data. We have proven that social data can be modelled in this way [3], which makes it a promsing path for phase 2.  

## Gradient Boosting as a Baseline

For xgboost, the most important parameters for the prediction task are:
1. `scale_pos_weight` to account for the unbalanced outcome.
2. `reg_lambda` for L2 regularization to avoid overfitting.

xgboost serves as a robust baseline with an `F1-score = 0.71` on the hidden validation set of round 2.

#### What Are the Important Questions?

The most important features/questions are displayed in Fig. 1. The main characteristics of these questions are:
1. `cf` code, indicating `Family & Household` type of questions.
2. `2020` most of the important questions are from the last year of the survey.

The most important question is `Within how many years do you hope to have your [first/next] child? (in 2020)`. This question remains significant across different survey years (cf. Fig. 1). This finding aligns with [2], where the author desmontrate that task-specific information outperforms surveillance-style big data for predicting academic performance. 

<figure>
  <img src="figures/feature_importance.png" alt="drawing" width="700"/>
  <figcaption><b>Fig. 1:</b> Feature importance from xgboost, with error bars representing the standard deviation on the feature importance across 50 runs.</figcaption>
</figure>

#### How Much Data/How Many Questions Do We Need?

xgboost requires only the top ~200 questions to achieve the best predictive performance. Performance plateaus and even decline when additional features are included (Fig. 2).

<figure>
  <img src="figures/scores.png" alt="drawing" width="700"/>
  <figcaption><b>Fig. 2:</b> Prediction metrics for an increasing number of features, ordered by feature importance (cf. Fig. 1). Each run is cross-validated over 50 80/20 train/test random splits.</figcaption>
</figure>

To mitigate overfitting, we employ cross-validation with an 80/20 split. Fig. 3 shows the distribution of `F1-score`. The mean is around 0.75, which is higher than the 0.71 on the validation set, indicating that the model still overfits despite the heavy regularization parameters.
<img src="figures/f1_score_distribution.png" alt="drawing" width="700"/>

## AutoEncoder

The AutoEncoder is designed to uncover complex, non-linear interactions between variables that traditional models might overlook. This approach allows us to:
1. Capture latent representations of the data.
2. Improve predictive performance by learning from the entire dataset structure.

The AutoEncoder is pretrained and finetuned in the following ways:

### Pretraining and finetuning
The Auto-Encoder is pretrained using standard reconstruction loss, where we use an **encoder** to compress the input and a **decoder** to decompress the compressed input and train the model to reconstruct the input. This is done using `CrossEntropyLoss` using the original input and the reconstructed input. 

For finetuning, we drop the decoder and work only with the encoded (compressed) input. Since we have 14 sequences per person, one for each year in the survey (detailed in Data Processing), we encode each sequence into a single vector, representing a highly compression representation of a survey. We then treat the 14 sequences, now compressed as as 14 vectors, as a new temporal sequence which is fed into a GRU.  

## Final Model

Our best performing model reuses the idea from the **AutoEncoder** pipeline. 
Here, we use *ExcelFormer* model [4] to create dense representation of input, i.e. answers to a questionnaire, followed by the RNN model that captures temporal interactions between questionnaires of different years. 
Finally, we use a simple attention mechanism to aggregate the output of the RNN and create a person-embedding.

We use these person embeddings to make the final predictions.

### ExcelFormer
The *ExcelFormer* is a transformer-based model designed for the *tabular* data (in our work, we redesign certain aspects of the pipeline):

1. Each column (question) and possible answer (category) are embedded into high dimensional space - we have separate embedding space for answers and columns (see Data Processing for more details),
2. *ExcelFormer* takes the sequence of the embedded questions-answers (that corresponds to a survey from a specific person $p$ for a specific year $y$) and passes it throught encoder layers.
3. The *ExcelFormer* is trained directly on the downstream task. 

Prior passing questionnaire-sequence to the model one need to sort the columns based on the importance to the target value (i.e. fertility). 
We take these importance scores from our experiments with the xgboost model.  

The main reason is the mechanism behind the *ExcelFormer* self-attention: the more important columns are not allowed to incorporate information from the less important columns (while the less important columns can do so). 

For each person $p$, we embed the corresponding surveys $s$ for each year. 
If person did not answer a survey for a specific year, we assume the embedding of the survey is a $\boldsymbol{0}$ vector. 

### RNN model (with attention mechanism)

A set of embedded surveys $S_p = \{s_0, s_1, s_2, .. s_{13}\}$ of a person $p$ is then passed to a RNN model in a chronological order. RNN returns a contextualized representation of a survey $s$ at a given timepoint - hence RNN still returns a set of surveys.

We aggregate these contextualized representations, $\mathbf{h}_t$, using a simple attention mechanism:

$$
a_t = \frac{e^{\mathbf{h}_t \cdot \mathbf{c}}}{\sum e^{\mathbf{h}_j \cdot \mathbf{c}}} \\
$$

Here, $c$ is a learnable context vector. The final person embedding is 
$$
\widehat{\mathbf{h}} = \sum a_t \cdot \mathbf{h}_t
$$

The two-year bidirectional GRU [5] provided the best results. 

### Limited Data

Since we have a limited number of labeled samples, we introduce the *Training with  Exponential Moving Average*: we take the weights of a model after each training epoch and average them based on 
$$
W_{t+1}^{\mathrm{EMA}}= 0.99 \cdot W_t^{\mathrm{EMA}}+ 0.01 \cdot W_t^{\text {model }}.
$$

We start averaging after the 3rd epoch, and the calculations are performed after each batch.

This setting lowers the chance of overfitting and provides better generalisability.

### Performance 

Based on our boostrap estimates (on the hold out dataset) the performance on the real unseen data is (with 95% confidence intervals):
  1. F1-Score: 0.771 [0.667, 0.861]
  2. MCC : 0.720 [0.600, 0.828]
  3. Precision: 0.895 [0.786, 0.976]
  4. Recall: 0.679 [0.545, 0.808]
  5. mean Averape Precision: 0.872 [0.789, 0.937]

After estimating the metric using our data splits, we retrained the model on the full dataset available. 

The full training pipeline is in the `train_finetune.ipynb` notebook. 

## Data Processing

We do a series of data processing steps to convert the tabular data into sequential data, to represent the temporal aspect of the surveys. We create a sequence for each year and tokenize each sequence. We decided to work only with categorical, numeric and date columns, as they are easy to process and represent most of the data. There also exists a file to handle free-text in `data_processing/text2vec`, but this requires an external model (LLM) to create the embeddings, which is why we have chosen not to include it.

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

# Conclusion 
We compared a common xgboost model with a more advanced machine learning model, which is still uncommon in the social sciences. The xgboost model provided a strong baseline, effectively identifying the most important factors influencing fertility outcomes. Additionally, we quantified the uncertainty of feature predictions using this model.

While the advanced model did not perform on par with xgboost in this phase of the challenge, it demonstrated potential for future improvements. This model has the capacity to be extended to richer registry data [3], which will be explored in phase two of the challenge. Its ability to capture complex, non-linear interactions holds promise for uncovering new insights and advancing our understanding of fertility behavior.


# References

[1] Sivak, E., Pankowska, P., Mendrik, A., Emery, T., Garcia-Bernardo, J., Hocuk, S., Karpinska, K., Maineri, A., Mulder, J., Nissim, M., & Stulp, G. (2024). Combining the Strengths of Dutch Survey and Register Data in a Data Challenge to Predict Fertility (PreFer) 

[2] Bjerre-Nielsen, A., Kassarnig, V., Lassen, D. D., & Lehmann, S. (2021). Task-specific information outperforms surveillance-style big data in predictive analytics. *PNAS*

[3] Savcisens, G., Eliassi-Rad, T., Hansen, L. K., Mortensen, L. H., Lilleholt, L., Rogers, A., Zettler, I., & Lehmann, S. (2024). Using sequences of life-events to predict human lives. *Nature Computational Science*

[4] Chen, J., Yan, J., Chen, D., Sun, J., & Wu, J. (2023). ExcelFormer: Making Neural Network Excel in Small Tabular Data Prediction. *arXiv preprint*

[5] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv*:1412.3555.