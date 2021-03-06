from __future__ import print_function

import argparse
import os
import pandas as pd
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


param_grid = {
 'alpha' : [0.01, 0.05, 0.10, 0.2, 0.5, 1.0, 2.0]
}


def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation-dir', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--alpha', type=float, default=0.01)
    
    args = parser.parse_args()

    training_dir = args.data_dir
    validation_dir = args.validation_dir
    
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    train_y = train_data.loc[:,0]
    train_x = train_data.loc[:,1:]

    mnb_classifier = MultinomialNB(alpha=args.alpha)
    
    model = GridSearchCV(
        estimator=mnb_classifier,
        scoring=['f1_macro'],
        refit='f1_macro',
        param_grid=param_grid,
        cv=10,
        verbose=1,
        n_jobs=1)
    
    model.fit(train_x, train_y)
    
    print("\nBest estimator: {0}".format(model.best_estimator_))
    print("\n")
    print(pd.DataFrame(model.cv_results_))
    
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    