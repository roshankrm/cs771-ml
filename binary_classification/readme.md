# ML Models for Emoticon, Text Sequence, Deep Features, and Combined Datasets

## Overview

This project implements various machine learning models to classify emoticon data, text sequence data, and deep feature data from multiple datasets. The models used are:

1. **Logistic Regression** for deep feature classification.
2. **Convolutional Neural Network (CNN) + GRU** for text sequence classification.
3. **Logistic Regression** for emoticon classification.
4. **Logistic Regression** for combined dataset classification (combining emoticon, text sequence, and deep feature datasets).

The code allows you to:
- Train the models on the datasets.
- Make predictions on test data.
- Save the predictions to text files.
- Print validation accuracies for each model.

## Project Structure

- **MLModelFeature**: Logistic Regression model for deep features dataset.
- **MLModelTextSeq**: CNN + GRU model for text sequence dataset.
- **MLModelEmoticon**: Simple Neural Network model for emoticon dataset.
- **MLModelCombined**: Logistic Regression model for the combined dataset (emoticon, text sequence, deep features).
- **TextSeqModel**: Inherits from MLModelTextSeq, preprocesses text sequence data, and trains the CNN + GRU model.
- **EmoticonModel**: Inherits from MLModelEmoticon, preprocesses emoticon data, and trains the neural network model.
- **FeatureModel**: Inherits from MLModelFeature, preprocesses deep feature data, applies PCA, and trains the logistic regression model.
- **CombinedModel**: Combines the emoticon, text sequence, and deep features into a unified dataset, applies PCA, and trains the logistic regression model.

## Requirements

- Python 3.x
- Required libraries:
  - numpy
  - pandas
  - scikit-learn
  - tensorflow
  - keras
  - matplotlib (optional, if you need visualizations)

You can install the required packages using:

```pip install -r requirements.txt```
## How to run the code

- Make sure all the dataset files are in the same folder as same as 2.py
- You can run the code using the command
```python3 2.py```

