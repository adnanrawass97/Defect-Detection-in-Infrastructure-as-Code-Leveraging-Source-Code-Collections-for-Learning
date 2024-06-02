# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
import numpy as np


from tqdm import tqdm, trange
import multiprocessing
from model import Model
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label


        
#def convert_examples_to_features(js,tokenizer,args):
    #source
    #max_token_length = 0
    #code=' '.join(js['code'].split())
    #code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    #source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    #source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    #padding_length = args.block_size - len(source_ids)
    #source_ids+=[tokenizer.pad_token_id]*padding_length
    #return InputFeatures(source_tokens,source_ids,js['label'])
def convert_examples_to_features(js, tokenizer, args):
    # source
    max_token_length = 0
    code = ' '.join(js['code'].split())
    window_size = args.block_size - 2  # Adjust window size as needed (consider max sequence length)

    # Handle cases where code length is less than window size
    if len(code.split()) < window_size:
        code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        return InputFeatures(source_tokens, source_ids, js['label'])

    # Windowing for long code snippets
    features = []
    for start_idx in range(0, len(code), window_size):

        end_idx = min(start_idx + window_size-2, len(code))
        window_code = code[start_idx:end_idx]
        code_tokens = tokenizer.tokenize(window_code)[:args.block_size - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = args.block_size - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        features.append(InputFeatures(source_tokens, source_ids, js['label']))

    return features  # Return a list of InputFeatures for each window

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    print("*** Example ***")
                    print("label: {}".format(example.label))
                    print("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    print("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    #def __getitem__(self, i):       
        #return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
    def __getitem__(self, i):
        if isinstance(self.examples[i], list):  # Check if it's a list of features
            # Access the first element (assuming one window per example)
            example = self.examples[i][0]
            return torch.tensor(example.input_ids), torch.tensor(example.label)
        else:
            # Code for handling single feature case (if applicable)
            return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)
        
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    patience = 10 
    """ Train the model """ 
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=8,pin_memory=True)
    

    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1,
                                                num_training_steps=max_steps)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  batch size = %d", args.train_batch_size)
    print("  Total optimization steps = %d", max_steps)
    best_acc=0.0
    model.zero_grad()
 
    for idx in range(args.num_train_epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        losses=[]
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx,round(np.mean(losses),3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
        print(len(tokenizer))        
        results = evaluate(args, model, tokenizer)
        for key, value in results.items():
            print("  %s = %s", key, round(value,4))    
            
        # Save model checkpoint
        if results['eval_acc']>best_acc:
            best_acc=results['eval_acc']
            patience = 10 
            print("  "+"*"*20)  
            print("  Best acc:%s",round(best_acc,4))
            print("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-acc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            print("Saving model checkpoint to %s", output_dir)
        else:
              patience -= 1
            # Early stopping condition
        if patience == 0:
             print("Early stopping triggered after", args.num_train_epochs - idx, "epochs with no improvement.")
             break
        
                        



def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits.argmax(-1)
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result

def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("***** Running Test *****")
    print("  Num examples = %d", len(eval_dataset))
    print("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits.argmax(-1)
    with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf-8") as f:
      for label in labels:
          f.write(str(label) + '\n')  # Write each label with a newline
    with open(os.path.join(args.output_dir,"predictions.txt"),'w', encoding="utf-8") as f:
        for example,pred in zip(eval_dataset.examples,preds):
            if pred:
                f.write('1 \n')
            else:
                f.write('0 \n')    
    
                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=42,
                        help="num_train_epochs")


    args = parser.parse_args()
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.n_gpu = torch.cuda.device_count()
   
    #if torch.backends.mps.is_available():
 
        #device = torch.device("mps")
        #args.n_gpu =1
    #else:
        #print("not avilable cuda ia habibe")
        #device = torch.device("cpu")
        #args.n_gpu =0

    
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels=100
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    vocab_size = len(tokenizer.get_vocab())
    print(f"Number of Embeddings (Vocabulary Size): {vocab_size}")
    config = RobertaConfig.from_pretrained("roberta-base")

    embedding_dim = config.hidden_size
    print(f"Embedding Dimension (Size): {embedding_dim}","before change")
  
    

    #vocab = tokenizer.get_vocab()

    #unique_tokens = set()
    #for token, index in vocab.items():
        #unique_tokens.add(token)
    
    #new_vocab = {token: index for index, token in enumerate(unique_tokens)}
    #tokenizer.vocab = new_vocab
    #tokenizer.model_max_length = tokenizer.model_max_length - (len(vocab) - len(new_vocab))
    #print("the max tokens number is",tokenizer.model_max_length)
 
    #print("the new number of tokens become ",len(new_vocab))
    model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,config=config)
    #model=AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    #vocab = tokenizer.get_vocab()


    model=Model(model,config,tokenizer,args)

    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    print("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args,args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))      
        model.to(args.device)
        result=evaluate(args, model, tokenizer)
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))                  
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()


