#*****************Note: keep all the dataset files in the same folder in which you are running this file***************************
# Import necessary libraries for neural networks, logistic regression, and data processing
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Embedding, Dense, Dropout, Conv1D, MaxPooling1D, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


class MLModelFeature():
    # Logistic Regression model for Dataset 2 (Deep Features)
    def __init__(self) -> None:
        # Initialize Logistic Regression with a max of 500 iterations
        self.model = LogisticRegression(random_state=42, max_iter=500)        
    
    def train(self, X_train, Y_train):
        # Train Logistic Regression model on training features and labels
        self.model.fit(X_train, Y_train)
        return self.model
        
    def predict(self, X):
        # Convert predictions to binary output (0 or 1)
        test_predictions = self.model.predict(X)
        test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
        return test_predictions_binary
        


class MLModelTextSeq():
    # CNN + GRU model for Dataset 3 (Text Sequence)
    def __init__(self) -> None:
        
        # Hyperparameters for the model
        vocab_size = 10
        embedding_dim = 8

        # Sequential model with embedding, Conv1D, GRU, and dense layers  
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
        self.model.add(Conv1D(filters=45, kernel_size=3, activation='relu'))
        self.model.add(BatchNormalization())# Normalizes the output of Conv1D for stable training
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(GRU(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))
        self.model.add(Dropout(0.5))# Regularization to prevent overfitting
        self.model.add(Dense(8, activation='relu'))# Intermediate fully-connected layer
        self.model.add(Dense(1, activation='sigmoid'))# Sigmoid for binary classification

        # Compile the model with Adam optimizer and binary cross-entropy loss
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self, X_train, y_train, X_val, y_val):
        # Early stopping to prevent overfitting if validation loss does not improve
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        # Train the model on training data with early stopping
        self.model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping],verbose=0)
        
    def predict(self, X):
        # Predict on new data
        test_predictions = self.model.predict(X)
        # Convert predictions to binary output (0 or 1)
        test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
        return test_predictions_binary
        
class MLModelEmoticon():
    # Simple Neural Network model for Dataset 1 (Emoticon Dataset)
    def __init__(self) -> None:
        # Build a simple sequential model with Embedding and Dense layers
        self.model = Sequential()
        self.model.add(Embedding(input_dim=215, output_dim=8))# Input is one-hot encoded emoticons
        self.model.add(Flatten())# Flatten the embeddings into a single vector
        self.model.add(Dense(1, activation='sigmoid'))# Sigmoid for binary classification
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])# Compile model
        
    def train(self, X_train, y_train, X_val, y_val):
        # Train the model on training data
        self.model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32,verbose=0)
    
    def predict(self, X):
        # Predict on new data
        test_predictions = self.model.predict(X)
        # Convert predictions to binary output (0 or 1)
        test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
        return test_predictions_binary


class MLModelCombined():
    # Logistic Regression model for the combined dataset (features from different datasets)
    def __init__(self) -> None:
        # Logistic Regression with 1000 iterations
        self.deep_model = LogisticRegression(max_iter=1000)
    
    def train(self, X_train, y_train):
        # Train the Logistic Regression model
        self.deep_model.fit(X_train, y_train)
    
    def predict(self, X):
        # Predict on new data
        test_predictions = self.deep_model.predict(X)
        # Convert predictions to binary output (0 or 1)
        test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
        return test_predictions_binary
    
class TextSeqModel(MLModelTextSeq):
    # TextSeqModel inherits from MLModelTextSeq and applies it to a specific dataset
    def __init__(self) -> None:
        super().__init__()
        # Load train and validation datasets for text sequences
        train_df = pd.read_csv('train_text_seq.csv')
        valid_df = pd.read_csv('valid_text_seq.csv')

        # Convert sequences to lists of integers
        train_sequences = train_df['input_str'].apply(lambda x: [int(char) for char in x]).tolist()
        valid_sequences = valid_df['input_str'].apply(lambda x: [int(char) for char in x]).tolist()

        # Extract labels
        y_train = train_df['label'].values
        y_val = valid_df['label'].values
        
        # Pad sequences to ensure uniform length
        max_sequence_length = 50
        X_train = [seq[:max_sequence_length] for seq in train_sequences]
        X_val = [seq[:max_sequence_length] for seq in valid_sequences]

        # Zero-pad sequences to ensure uniform input size
        X_train = [seq + [0] * (max_sequence_length - len(seq)) if len(seq) < max_sequence_length else seq for seq in X_train]
        X_val = [seq + [0] * (max_sequence_length - len(seq)) if len(seq) < max_sequence_length else seq for seq in X_val]

        X_train = np.array(X_train)
        X_val = np.array(X_val)

        # Train the model
        self.train(X_train, y_train, X_val, y_val)

    def predict(self, X):
        # Preprocess and predict on new text sequences
        test_sequences = X['input_str'].apply(lambda x: [int(char) for char in x]).tolist()
        max_sequence_length = 50
        
        X_test =  [seq[:max_sequence_length] for seq in test_sequences]
        
        X_test = [seq + [0] * (max_sequence_length - len(seq)) if len(seq) < max_sequence_length else seq for seq in X_test]
        
        X_test = np.array(X_test)
        
        test_predictions = super().predict(X_test)
        return test_predictions
    
    
class EmoticonModel(MLModelEmoticon):
    # EmoticonModel inherits from MLModelEmoticon and applies it to a specific dataset
    def __init__(self) -> None:
        super().__init__()
        # Load train and validation datasets for emoticons
        train_df = pd.read_csv("train_emoticon.csv")
        valid_df = pd.read_csv("valid_emoticon.csv")
        X_train = train_df['input_emoticon']
        y_train = train_df['label']
        X_valid = valid_df['input_emoticon']
        y_valid = valid_df['label']

        # Tokenize emoticons into sequences
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_valid_seq = self.tokenizer.texts_to_sequences(X_valid)
        
        # Pad sequences to ensure uniform input size
        max_len = 13
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
        X_valid_pad = pad_sequences(X_valid_seq, maxlen=max_len, padding='post')
        
        # Train the model
        self.train(X_train_pad, y_train, X_valid_pad, y_valid)
        
    def predict(self, X):
        # Preprocess and predict on new emoticons
        X_test_seq = self.tokenizer.texts_to_sequences(X)
        X_test_pad = pad_sequences(X_test_seq, maxlen=13, padding='post')
        test_predictions = super().predict(X_test_pad)
        return test_predictions
        
    
class FeatureModel(MLModelFeature):
    # Logistic Regression for Deep Features dataset (Dataset 2)
    def __init__(self) -> None:
        super().__init__()
        
        # Load training data for deep features
        train_feat = np.load("train_feature.npz", allow_pickle=True)
        X = train_feat['features']
        Y = train_feat['label'].astype(int)
        
        # Flatten and apply PCA for dimensionality reduction
        X_flatten = X.reshape(X.shape[0], -1)
        self.pca = PCA(n_components=194)
        X_pca = self.pca.fit_transform(X_flatten)

        # Train the Logistic Regression model
        self.model=self.train(X_pca, Y)


    def predict(self, X):
        # Predict on new deep features using PCA
        X_test_flatten = X.reshape(X.shape[0], -1)
        X_test_pca = self.pca.transform(X_test_flatten)
        test_predictions = super().predict(X_test_pca)
        test_predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]
        return test_predictions_binary
        
    
class CombinedModel(MLModelCombined):
    # CombinedModel for combining features from multiple datasets
    def __init__(self) -> None:
        super().__init__()
        
        # Load and preprocess emoticon, deep feature, and text sequence datasets
        train_emoticon_data = pd.read_csv('train_emoticon.csv')
        X_emoticon = train_emoticon_data['input_emoticon'].apply(lambda x: [ord(emoticon) % 10 for emoticon in x]).tolist()
        X_emoticon = pad_sequences(X_emoticon, maxlen=13, padding='post')
        
        train_deep_features = np.load('train_feature.npz')
        X_deep = train_deep_features['features'].reshape(train_deep_features['features'].shape[0], -1)
        
        train_text_seq_data = pd.read_csv('train_text_seq.csv')
        X_seq = train_text_seq_data['input_str'].apply(lambda x: [int(char) for char in x]).tolist()
        X_seq = pad_sequences(X_seq, maxlen=50, padding='post')
        
        y_train = train_emoticon_data['label']

        # Combine the three sets of features
        X_combined_train = np.hstack([X_emoticon, X_deep, X_seq])

        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=250)
        X_train_pca = self.pca.fit_transform(X_combined_train)
        
        # Train the Logistic Regression model on the combined features
        self.train(X_train_pca, y_train)

    def predict(self, test_feat_X, test_emoticon_X, test_seq_X):        
        # Preprocess and predict on new combined test data
        X_deep_test = test_feat_X.reshape(test_feat_X.shape[0],-1)
        
        X_emoticon_test = test_emoticon_X.apply(lambda x: [ord(emoticon) % 10 for emoticon in x]).tolist()
        X_emoticon_test = pad_sequences(X_emoticon_test, maxlen=13, padding='post')
        
        X_seq_test = test_seq_X['input_str'].apply(lambda x: [int(char) for char in x]).tolist()
        X_seq_test = pad_sequences(X_seq_test, maxlen=50, padding='post')
        
        # Combine the test datasets
        X_combined_test = np.hstack([X_emoticon_test, X_deep_test, X_seq_test])
        
        # Apply PCA and predict on the combined features
        X_test_pca = self.pca.transform(X_combined_test)
        test_predictions = super().predict(X_test_pca)
        return test_predictions
    
    
def save_predictions_to_file(predictions, filename):
    # Function to save predictions to a file
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # Load test and validation datasets for different features (emoticon, deep features, text sequences)
    test_feat_X = np.load("test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("test_emoticon.csv")['input_emoticon']
    test_seq_X = pd.read_csv("test_text_seq.csv")

    valid_emoticon_data = pd.read_csv('valid_emoticon.csv')
    valid_deep_features = np.load('valid_feature.npz', allow_pickle=True)['features']
    valid_text_seq_data = pd.read_csv('valid_text_seq.csv')
    y_valid = valid_emoticon_data['label']
    
    # Initialize Emoticon model and save predictions
    emoticon_model  = EmoticonModel()
    pred_emoticon_valid = emoticon_model.predict(valid_emoticon_data['input_emoticon'])
    print("Validation Accuracy on Emoticon Dataset - ",accuracy_score(y_valid,pred_emoticon_valid))
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    
    # Initialize Feature model and save predictions
    feature_model = FeatureModel()
    pred_feat_valid = feature_model.predict(valid_deep_features)
    print("Validation Accuracy on Features Dataset - ",accuracy_score(y_valid,pred_feat_valid))
    pred_feat = feature_model.predict(test_feat_X)
    save_predictions_to_file(pred_feat, "pred_deepfeat.txt")
    
    # Initialize Text Sequence model and save predictions
    text_model = TextSeqModel()
    pred_text_valid = text_model.predict(valid_text_seq_data)
    print("Validation Accuracy on Text Sequence Dataset - ",accuracy_score(y_valid,pred_text_valid))
    pred_text = text_model.predict(test_seq_X)
    save_predictions_to_file(pred_text, "pred_textseq.txt")
    
    # Initialize Combined model and save predictions
    Combined_model = CombinedModel()
    pred_combined_valid = Combined_model.predict(valid_deep_features, valid_emoticon_data['input_emoticon'], valid_text_seq_data)
    print("Validation Accuracy on combined Dataset - ",accuracy_score(y_valid, pred_combined_valid))
    pred_combined = Combined_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    save_predictions_to_file(pred_combined, "pred_combined.txt")
