BERT-Based Text Classification for Racial Comment Detection
This repository contains code for training a BERT-based model to classify comments into racial categories using textual data. The model is designed to predict the racial nature of a comment (i.e., offensive, neutral, etc.) based on user-generated text, leveraging the power of transformers and deep learning.

Requirements
To run this code, ensure you have the following libraries installed:

pandas
numpy
torch
transformers
sklearn
matplotlib
You can install the required libraries using pip:

bash
Copy
Edit
pip install pandas numpy torch transformers scikit-learn matplotlib
File Structure
bert_test.csv: Dataset containing comments with a "Racial" label (1 or 0) for each comment. The file must have two columns:

Comment_Text: The textual content of the comment.
Racial: The label indicating if the comment is racial in nature (e.g., 0 for non-racial, 1 for racial).
model.py: Main file where BERT is implemented and trained for the classification task.

README.md: This file containing information about the repository.

How to Use
1. Dataset
The dataset bert_test.csv should have the following structure:

Comment_Text	Racial
"This is a normal comment."	0
"This comment is racially offensive."	1
Make sure your dataset is formatted properly before running the model training.

2. Data Preprocessing
The train_test_split function splits the dataset into training, validation, and test sets with a 70%/15%/15% ratio, ensuring the class distribution is preserved (stratified split).

The comments are tokenized and padded to a fixed length using the BERT tokenizer (BertTokenizerFast).

3. Model Architecture
The model is based on BERT (Bidirectional Encoder Representations from Transformers), which is fine-tuned on the dataset for a text classification task. It includes:

A BERT layer (frozen parameters from the pre-trained bert-base-uncased model).
A fully connected layer (Linear) that maps the output to the target labels.
A ReLU activation function for non-linearity.
A Dropout layer for regularization to avoid overfitting.
A Softmax layer for multi-class classification (although this can be adapted for binary classification).
4. Training
The model is trained with cross-entropy loss (with class weights) and AdamW optimizer from Hugging Face’s transformers library.
A gradient clipping technique is used to handle exploding gradients during training.
The training process includes 10 epochs, with training and validation loss printed at each epoch.
The best model (based on validation loss) is saved in the file saved_weights.pt.
5. Evaluation
After training, the best model is used to evaluate the test set. The model’s performance is reported using classification_report from sklearn, which includes metrics such as Precision, Recall, F1-score, and Accuracy.

6. Example Usage
Run the following code to train and evaluate the model:

python
Copy
Edit
import torch
from model import BERT_Arch, train, evaluate

# Define your model
model = BERT_Arch(bert)

# Train the model
train()

# Evaluate the model
evaluate()

# Get the final predictions
predictions = model(test_seq, test_mask)
7. Hyperparameters
Batch size: 32
Learning rate: 1e-5
Number of epochs: 10
Feel free to adjust these parameters as needed for different datasets or training configurations.

Results
After training, the model will output the training loss and validation loss for each epoch. After completion, it will also print the performance metrics for the test set.

Here is an example of the classification_report output after testing:

markdown
Copy
Edit
              precision    recall  f1-score   support

           0       0.85      0.92      0.88       100
           1       0.79      0.67      0.73        50

    accuracy                           0.83       150
   macro avg       0.82      0.79      0.80       150
weighted avg       0.83      0.83      0.83       150
License
This project is licensed under the MIT License.
