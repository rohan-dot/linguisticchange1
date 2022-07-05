#!/scratch/u/erho/rhoenv/bin/python

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForMaskedLM
from torch.utils.data import Dataset, TensorDataset, RandomSampler, SequentialSampler
import torch
from transformers import Trainer, DataCollatorForLanguageModeling, TrainingArguments
from torch.utils.data import DataLoader
import scipy
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr import visualization as viz
from collections import defaultdict
import torch


CUDA = (torch.cuda.device_count() > 0)


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, input_ids, attn_mask, labels=None):
    self.input_ids = input_ids
    self.attn_mask = attn_mask
    if labels is not None:
        self.labels = labels

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, index):
    item = {
        'input_ids': self.input_ids[index],
        'attention_mask': self.attn_mask[index]
    }   
    if hasattr(self, 'labels'):
        item['labels'] = self.labels[index]

    return item


def build_dataset(texts, labels=None):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased', do_lower_case=True)

    tokenized_data = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        max_length=150,
        return_attention_mask=True)

    dataset = MyDataset(
        tokenized_data['input_ids'], 
        tokenized_data['attention_mask'],
        labels)

    return dataset, tokenizer


def compute_metrics(pred):
    """p: EvalPrediction obj"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def finetune_model(model, train_dataset, tokenizer, working_dir,
        mlm=False, epochs=3):
    if mlm:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm_probability=0.15)
    else:
        data_collator = None

    training_args = TrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        output_dir=working_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator)

    train_result = trainer.train()

    # works but removing in favor of classification_eval() below for AUC support
    # if eval_dataset is not None:
    #     eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    return train_result, model



def binary_classification_report(y_true, y_prob):
    y_prob = np.array(y_prob)
    y_hat = (y_prob > 0.5).astype('int')

    return {
        'confusion-matrix': metrics.confusion_matrix(y_true, y_hat).astype(np.float32),
        'precision': metrics.precision_score(y_true, y_hat),
        'recall': metrics.recall_score(y_true, y_hat),
        'f1': metrics.f1_score(y_true, y_hat),
        'acc': metrics.accuracy_score(y_true, y_hat),
        'auc': metrics.roc_auc_score(y_true, y_prob),
        # 'loss': metrics.log_loss(y_true, y_prob)
    }





def transfer_parameters(from_model, to_model):
    to_dict = to_model.state_dict()
    from_dict = {k: v for k, v in from_model.state_dict().items() if k in to_dict}
    to_dict.update(from_dict)
    to_model.load_state_dict(to_dict)

    return to_model





class Attributor:
    def __init__(self, model, target_class, tokenizer):
        """ TODO generalize to multiclass """
        self.model = model
        self.target_class = target_class
        self.tokenizer = tokenizer
        
        self.fwd_fn = self.build_forward_fn(target_class)

        self.lig = LayerIntegratedGradients(self.fwd_fn, self.model.distilbert.embeddings)


    def attribute(self, input_ids):

        ref_ids = [[x if x in [101, 102] else 0 for x in input_ids[0]]]

        attribution, delta = self.lig.attribute(
                inputs=torch.tensor(input_ids).cuda() if CUDA else torch.tensor(input_ids),
                baselines=torch.tensor(ref_ids).cuda() if CUDA else torch.tensor(ref_ids),
                n_steps=25,
                internal_batch_size=5,
                return_convergence_delta=True)

        attribution_sum = self.summarize(attribution)        

        return attribution_sum, delta

    def attr_and_visualize(self, input_ids, label):
        attr_sum, delta = self.attribute(input_ids)
        y_prob = self.fwd_fn(input_ids)
        pred_class = 1 if y_prob.data[0] > 0.5 else 0

        if CUDA:
            input_ids = input_ids.cpu().numpy()[0]
            label = label.cpu().item()
            attr_sum = attr_sum.cpu().numpy()
            y_prob = y_prob.cpu().item()
        else:
            input_ids = input_ids.numpy()[0]
            label = label.item()
            attr_sum = attr_sum.numpy()
            y_prob = y_prob.item()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        record = viz.VisualizationDataRecord(
            attr_sum, 
            y_prob,
            pred_class,
            label,
            self.target_class,
            attr_sum.sum(),
            tokens,
            delta)

        tok2attr = defaultdict(list)
        for tok, attr in zip(tokens, attr_sum):
            tok2attr[tok].append(attr)

        html = viz.visualize_text([record])

        return html.data, tok2attr, attr_sum, y_prob, pred_class


    def build_forward_fn(self, label_dim):

        def custom_forward(inputs):
            preds = self.model(inputs)[0]
            return torch.softmax(preds, dim=1)[:, label_dim]

        return custom_forward

    def summarize(self, attributions):
        """ sum across each embedding dim and normalize """
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        return attributions



def eval(model, dataset, tokenizer):
    dataloader = DataLoader(dataset, batch_size=1, # has to be batch size 1 
        sampler=SequentialSampler(dataset),
        # group by datatype
        collate_fn=lambda data: {k: [x[k] for x in data] for k in data[0].keys()})

    model.eval()

    if CUDA:
        model = model.cuda()

    attributor = Attributor(model, target_class=1, tokenizer=tokenizer)

    y_probs = []
    y_hats = []
    labels = []
    vizs = []
    tok2attr = None

    for batch in tqdm(dataloader):

        labels += batch['labels']

        if CUDA:
            batch = {k: torch.tensor(v).cuda() for k, v in batch.items()}
        else:
            batch = {k: torch.tensor(v) for k, v in batch.items()}

        viz, t2a, attrs, y_prob, y_hat = attributor.attr_and_visualize(
            batch['input_ids'], batch['labels'])

        if tok2attr is None:
            tok2attr = t2a
        else:
            for k, v in t2a.items():
                tok2attr[k] += v

        y_probs += [y_prob]
        y_hats += [y_hat]
        vizs.append(viz)

    report = binary_classification_report(labels, y_probs)
    return report, vizs, tok2attr


