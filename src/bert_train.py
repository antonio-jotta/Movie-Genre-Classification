import torch
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datasets import Dataset
import json
from transformers import EarlyStoppingCallback

def train_model(train, validation):
    # Prepare the training data
    X_train = train['Plot'].tolist()
    y_train = train['Genre'].tolist()
    directors_train = train['Director'].tolist()

    # Prepare the validation data
    X_val = validation['Plot'].tolist()
    y_val = validation['Genre'].tolist()
    directors_val = validation['Director'].tolist()

    # Encode genres
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[DIRECTOR]']})

    def tokenize_function(examples):
        director_names = [f"[DIRECTOR] {director}" for director in examples['director']]
        texts_with_directors = [f"{director_name} {text}" for director_name, text in zip(director_names, examples['text'])]
        return tokenizer(texts_with_directors, padding='max_length', truncation=True, max_length=512)

    # Prepare the datasets for Hugging Face
    train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train_encoded, 'director': directors_train})
    val_dataset = Dataset.from_dict({'text': X_val, 'label': y_val_encoded, 'director': directors_val})

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define a model class without the director embedding
    class DistilBertForGenreClassification(torch.nn.Module):
        def __init__(self, num_labels):
            super(DistilBertForGenreClassification, self).__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(self.bert.config.hidden_size, num_labels)
            )
            self.loss_fn = torch.nn.CrossEntropyLoss()

        def forward(self, input_ids, attention_mask, labels=None):
            # Get BERT output
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_outputs.last_hidden_state[:, 0, :]  # CLS token output
            # Get logits from classifier
            logits = self.classifier(pooled_output)
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}

    # Instantiate the model
    num_labels = len(label_encoder.classes_)
    model = DistilBertForGenreClassification(num_labels=num_labels)
    model.bert.resize_token_embeddings(len(tokenizer))

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=10,
        weight_decay=0.1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
    )

    # Define compute_metrics function for evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}

    # Data collator
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()
    return model
