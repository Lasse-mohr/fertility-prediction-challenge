import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

plot = False

if __name__ == '__main__':
    name = 'louis' 
    df_scores = pd.DataFrame(columns=['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1 Score','n_features'])
    threshold = 0.11
    X = pd.read_csv(f'training_data/PreFer_train_data_{name}_{threshold:.2f}.csv')
    y = pd.read_csv(f'training_data/PreFer_train_outcome_{name}_{threshold:.2f}.csv')
        
    #X = data.drop(['nomem_encr', 'new_child'], axis=1)
    #y = data['new_child']

    n= 40 #cross validation
    scores = np.zeros((n, 4))
    features_importance = np.zeros(X.shape[1])
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(100))
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        # Initialize the XGBoost classifier
        model = xgb.XGBClassifier(
            objective='binary:logistic', 
            use_label_encoder=False, 
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight, 
            max_delta_step=1,
            verbosity=0,
            tree_method='exact',
            learning_rate=0.1,
            subsample=0.8,
            reg_lambda = 5, #10
            reg_alpha = 0.1, #0.1
            )
        
        model.fit(X_train, y_train, verbose=0)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        features_importance += model.feature_importances_/n
        
        scores[i] = [accuracy, precision, recall, f1]
        
        cols = X.columns
        top_features = np.argsort(features_importance)[::-1][:25]
        top_scores = features_importance[top_features]
        top_cols = cols[top_features]
        
    accuracy = np.mean([score[0] for score in scores])
    precision = np.mean([score[1] for score in scores])
    recall = np.mean([score[2] for score in scores])
    f1 = np.mean([score[3] for score in scores])
    print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}")
    
    if plot:
        #histogram of f1 scores
        bins = np.linspace(0.5, 1, 30)
        h,b = np.histogram([score[3] for score in scores], bins=bins)
        fig, ax = plt.subplots()
        plt.bar(b[:-1], h, width=b[1]-b[0], color='teal', edgecolor='black', linewidth=1.2, alpha=0.7)
        plt.title('F1 Score Distribution', fontsize=16)
        plt.xlabel('F1 Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('figures/f1_score_distribution.png')
        
    
        #feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(top_cols, top_scores, color='teal', edgecolor='black', linewidth=1.2, alpha=0.7)
        plt.title('Feature Importance', fontsize=16)
        plt.tick_params(axis='x', which='major', pad=0)  
        plt.xticks(rotation=30)
        plt.tight_layout(pad=1.0)  
        fig.subplots_adjust(bottom=0.2) 
        #remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig('figures/feature_importance.png')