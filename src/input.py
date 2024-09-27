#%%
import pandas

train = pandas.read_csv("../data/train.txt", delimiter='\t', names=["Title", "Industry", "Genre", "Director", "Plot"])
test = pandas.read_csv("../data/test_no_labels.txt", delimiter='\t', names=["Title", "Industry", "Director", "Plot"])

train.isna().sum() # no treatment for MVs

X_train = train['Plot'].tolist()
y_train = train['Genre'].tolist()

X_test = test['Plot'].tolist()

from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

# Encode the Genre labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the plots for BERT input
def tokenize_plots(plots):
    return tokenizer(
        plots,
        padding=True,
        truncation=True,
        max_length=128,  # Set a max length for efficiency
        return_tensors='tf'
    )

# Tokenize the training plots
train_encodings = tokenize_plots(X_train)
# %%
