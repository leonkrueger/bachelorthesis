import os

import pandas as pd
from transformers.integrations import WandbCallback


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))
    return {"labels": labels, "predictions": prediction_text}


def compute_metrics(predictions) -> dict[str, float]:
    labels = predictions.label_ids
    preds = predictions.predictions.argmax(-1)

    accuracy = len([pred for pred, label in zip(preds, labels) if pred == label]) / len(
        preds
    )
    return {"accuracy": accuracy}


class WandbTablePredictionAccuracyCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, validation_dataset, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Control the frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        if state.eval_steps % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.validation_dataset)
            # decode predictions and labels
            accuracy = compute_metrics(decode_predictions(self.tokenizer, predictions))
            # add predictions to a wandb.Table
            # predictions_df = pd.DataFrame(predictions)
            # predictions_df["step"] = state.global_step
            # records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"accuracy": accuracy}, step=state.global_step)
