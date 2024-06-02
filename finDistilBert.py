from utils import preprocess, unfreezer
from transformers import (
    LineByLineTextDataset,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TFDistilBertForSequenceClassification,
    )
from sklearn.model_selection import train_test_split
from keras.optimizers.schedules import PolynomialDecay
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
import pandas as pd
import numpy as np
import os

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


class Config(object):
    """The configuration class for training."""

    def __init__(
            self,
            data_dir,
            distilbert_mask,
            model_dir,
            log_dir,
            # for tokenization
            max_seq_length=128,
            # for the further training
            further_train_batch_size=128,
            furtherTrain_lr=2e-5,
            furtherTrain_epochs=2,
            # for the fine tuning
            train_batch_size=64,
            eval_batch_size=64,
            fineTune_lr=2e-5,
            fineTune_epochs=6,
            seed=42,
            # for different training techniques
            discriminate=True,
            gradual_unfreeze=False,
            base_model="bert-base-uncased",
            ):
        self.data_dir = data_dir
        self.distilbert_mask = distilbert_mask
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.max_seq_length = max_seq_length
        self.further_train_batch_size = further_train_batch_size
        self.furtherTrain_lr = furtherTrain_lr
        self.furtherTrain_epochs = furtherTrain_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.fineTune_lr = fineTune_lr
        self.fineTune_epochs = fineTune_epochs
        self.seed = seed
        self.discriminate = discriminate
        self.gradual_unfreeze = gradual_unfreeze
        self.base_model = base_model


class FinDistilBert(object):
    """
    The main class for FinBERT.
    """

    def __init__(self, config):
        self.config = config

    def get_data(self, phrase):
        """
        Get the data for training the model:
        phrase: Either `pre-train` or `fine-tune`
        """
        # if during the pre-train process, return the PyTorch dataset object
        if phrase == "pre-train":
            # read the data
            file_path = os.path.join(self.config.data_dir, "raw.csv")
            raw_dataset = pd.read_csv(file_path)[["body", "sentiment"]]
            raw_dataset["body"] = raw_dataset["body"].apply(preprocess)
            raw_dataset["word_count"] = raw_dataset["body"].apply(
                lambda x: len(x.split())
                )
            raw_dataset = raw_dataset.loc[
                (raw_dataset.word_count >= 9) & (raw_dataset.word_count <= 130),
            ]
            raw_dataset = raw_dataset.drop_duplicates().reset_index(drop=True)
            text_file_path = os.path.join(self.config.data_dir, "financial_corpus.txt")
            raw_dataset["body"].to_csv(
                text_file_path,
                index=False,
                header=False,
                )
            dataset = LineByLineTextDataset(
                tokenizer=tokenizer,
                file_path=text_file_path,
                block_size=128,  # Can be adjusted due to my own requirement
                )
            return dataset

        elif phrase == "fine-tune":
            file_path = os.path.join(self.config.data_dir, "fpb.csv")
            fpb_dataset = pd.read_csv(file_path)
            X = list(fpb_dataset["content"])

            # 0 == 'neutral' ; 1 == "positive" ; 2 == "negative"
            y = list(fpb_dataset["sentiment"])
            y = tf.keras.utils.to_categorical(y, num_classes=3)

            # split the train and test dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.40, random_state=self.config.seed, stratify=y
                )

            # split the test and validation dataset
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_test,
                y_test,
                test_size=0.50,
                random_state=self.config.seed,
                stratify=y_test,
                )

            # preserve the y_test so it can be used in the evaluation process
            self.y_test = y_test

            train_encodings = tokenizer(
                X_train,
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
                )

            valid_encodings = tokenizer(
                X_valid,
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
                )

            test_encodings = tokenizer(
                X_test,
                truncation=True,
                padding=True,
                max_length=self.config.max_seq_length,
                )

            # convert the tokenized training dataset to tensors
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (dict(train_encodings), y_train)
                ).batch(self.config.train_batch_size)

            # convert the tokenized validation dataset to tensors
            valid_dataset = tf.data.Dataset.from_tensor_slices(
                (dict(valid_encodings), y_valid)
                ).batch(self.config.eval_batch_size)

            # convert the tokenized test dataset to tensors
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (dict(test_encodings), y_test)
                ).batch(self.config.eval_batch_size)

            return train_dataset, valid_dataset, test_dataset

    def further_train(self):
        # get the data
        dataset = self.get_data(phrase="pre-train")
        # do the masked language modelling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
        # set the training arguments
        training_args = TrainingArguments(
            output_dir=self.config.log_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.furtherTrain_epochs,
            per_device_train_batch_size=self.config.train_batch_size,  # Adjust based on your GPU
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=self.config.furtherTrain_epochs,  # specified learning rate
            )
        # setup the trainer
        trainer = Trainer(
            model=self.config.distilbert_mask,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            )

        trainer.train()
        save_path = os.pth.join(self.config.model_dir, "FurtherTrain")
        model.save_pretrained(self.save_path)

    def fine_tune(self, from_further_trained=True):
        # get the data for training and validation
        train_dataset, valid_dataset, _ = self.get_data(phrase="fine-tune")
        num_train_steps = len(train_dataset) * self.config.fineTune_epochs

        # set the learning rate scheduler
        lr_scheduler = PolynomialDecay(
            initial_learning_rate=6e-5,
            end_learning_rate=3e-5,
            decay_steps=num_train_steps,
            )

        # set the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

        # set the loss function
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        if from_further_trained:
            model_path = os.path.join(self.config.model_dir, "FurtherTrain")
            self.model = TFDistilBertForSequenceClassification.from_pretrained(
                model_path, num_labels=3
                )
            save_path = os.path.join(self.config.model_dir, "FurtherTrain+FineTune")
        else:
            self.model = TFDistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
                )
            save_path = os.path.join(self.config.model_dir, "FineTune")

        # in this paper, I didn't perform gradual_unfreezing technique, however, I provided the option
        ########################################
        ####   without gradual unfreezing   ####
        ########################################
        if not self.config.gradual_unfreeze:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

            self.fine_tune_history = self.model.fit(
                train_dataset,
                epochs=self.config.fineTune_epochs,
                validation_data=valid_dataset,
                )
            self.model.save_pretrained(save_path)
            return self.fine_tune_history
        ########################################
        ####     with gradual unfreezing    ####
        ########################################
        elif self.config.gradual_unfreeze:
            for layer in self.model.layers:
                layer.trainable = False

            self.history = dict()
            training_loss = []
            validation_loss = []

            for epoch in range(self.config.fineTune_epochs):
                for step, batch in enumerate(train_dataset):
                    unfreezer(epoch, step, self.model)
                    with tf.GradientTape() as tape:
                        inputs, labels = batch
                        outputs = self.model(inputs, training=True).logits
                        loss_value = loss(labels, outputs)

                    if step % 50 == 0:
                        print(
                            "Epoch: {}, Step: {}, Loss: {:.5f}.".format(
                                epoch, step, loss_value.numpy()
                                )
                            )

                    # record the loss for futher plotting and analysis

                    training_loss.append(loss_value.numpy())
                    print("Training Loss: {}".format(loss_value.numpy()))

                    grads = tape.gradient(loss_value, self.model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(grads, self.model.trainable_variables)
                        )

                # Initialize variables to calculate average loss
                total_val_loss = 0

                # Iterate over the dataset to calculate the loss
                for val_batch, val_labels in valid_dataset:
                    # Use your model's loss function, assuming it's 'loss'
                    val_predictions = self.model.predict(val_batch, verbose=0).logits
                    val_loss_value = loss(val_labels, val_predictions)

                    total_val_loss += val_loss_value.numpy()

                # Calculate average validation loss
                avg_val_loss = total_val_loss / len(test_dataset)
                validation_loss.append(avg_val_loss)
                print("In Epoch {}, Validation Loss is {}".format(epoch, avg_val_loss))

            self.history["loss"] = training_loss
            self.history["val_loss"] = validation_loss
            return self.history

    def evaluate(self):
        # define the loss
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.metrics = dict()
        # get the data necessary
        _, _, test_dataset = self.get_data(phrase="fine-tune")
        y_prediction = self.model.predict(test_dataset).logits
        # get the CategoricalCrossEntropy Loss
        self.metrics["loss"] = loss(self.y_test, y_prediction).numpy()
        # now convert them to 1D array
        y_prediction = y_prediction.argmax(axis=1)
        self.y_test = self.y_test.argmax(axis=1)
        self.metrics["accuracy"] = accuracy_score(self.y_test, y_prediction)
        self.metrics["cm"] = confusion_matrix(self.y_test, y_prediction)
        self.metrics["f1"] = f1_score(self.y_test, y_prediction, average="macro")
        return self.metrics, y_prediction
