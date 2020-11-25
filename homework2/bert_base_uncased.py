import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer 
from transformers import BertForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split 

from google.colab import drive
drive.mount('/content/drive') 
os.chdir('/content/drive/MyDrive/Colab')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, output_attentions=False, output_hidden_states=False )

def data_clean(raw_train, raw_test):
    # clean raw train and raw test data
    train_txt = raw_train.readlines()
    train = [[train_txt[3*(i-1)][:-1], train_txt[3*(i-1)+1][:-1], train_txt[3*(i-1)+2][:2]] for i in range(int(len(train_txt)/3))]
    test_txt = raw_test.readlines()
    test = [[test_txt[2*(i-1)][:-1], test_txt[2*(i-1)+1][:-1]] for i in range(int(len(test_txt)/2))]
    pattern = r"\$T\$"
    pattern = re.compile(pattern)
    # modify the training data to dataframe
    same_count = 0
    for it in train:
        key = it[0]
        pattern = re.compile(pattern)
        newKey = re.sub(pattern, it[1], key)
        it[0] = newKey
        it.remove(it[1])
    train_id = range(len(train))
    train_txt = [train[i][0] for i in range(len(train))]
    train_label = [(int(train[i][1])+1) for i in range(len(train))]
    df = {"id": train_id, "text": train_txt, "label": train_label}
    train = pd.DataFrame(df)
    print('Number of train sentences: {:,}\n'.format(train.shape[0]))

    # modify the test data to dataframe
    for it in test:
        key = it[0]
        newKey = re.sub(pattern, it[1], key)
        it[0] = newKey
        it.remove(it[1])
    test_id = range(len(test))
    test_txt = [test[i][0] for i in range(len(test))]
    test_label = [0 for i in range(len(test))]
    df = {"id": test_id, "text": test_txt, "label": test_label}
    test = pd.DataFrame(df)
    print('Number of test sentences: {:,}\n'.format(test.shape[0]))

    return train, test


def evaluate(dataset_val):
    predictions, true_label = [], []
    dataloader_val = DataLoader(dataset_val, sampler = RandomSampler(dataset_val), batch_size= 32)
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        label = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_label.append(label)

    predictions = np.concatenate(predictions, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    i = correct = total = 0
    for result in predictions:
        if (np.argmax(result) - (int)(np.argmax(result) == 0)) == int(true_label[i]):
            correct += 1
        total += 1
        i += 1
    acc_val = correct / total
    return acc_val

def train_Bert(train):
    # loading tokenizer and encoding training data and test data
    # training data
    d_train = train[train.data_type=='train']
    d_val = train[train.data_type=='val']
    encoded_data_train = tokenizer.batch_encode_plus(d_train.text.values,
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    padding='longest',
                                                    max_length=256,
                                                    return_tensors='pt'
                                                    )
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(d_train.label.values)
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

    encoded_data_val = tokenizer.batch_encode_plus(d_val.text.values,
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    pad_to_max_length=True,
                                                    max_length=256,
                                                    return_tensors='pt'
                                                    )
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(d_val.label.values)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    # set-up pre-trained Bert model, optimizers, and creating data loaders
    # setting model
    model.to(device)
    # creating data loaders
    dataloader_train = DataLoader(dataset_train,sampler = RandomSampler(dataset_train), batch_size= 32)
    # setting optimizers
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,eps=1e-8)
    epochs = 20
    num_epoch = 0
    acc_val_list = []
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)

    for epoch in tqdm(range(1, epochs+1)):
        num_epoch += 1
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(dataloader_train, 
                            desc ='Epoch {:1d}'.format(epoch),
                            leave=False,
                            disable=False
                            )
        for batch in progress_bar:
            model.zero_grad()
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'labels' : batch[2]
                    }
            outputs = model(**inputs)

            loss = outputs[0]
            print(loss.item())
            loss_train_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
        
        # printing calues after each epoch
        torch.save(model.state_dict(), f'BERT_ft_epoch.model')
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        # calculate acc on train and val
        acc_val = evaluate(dataset_val)
        tqdm.write(f'Accuracy on val: {acc_val}')
        acc_val_list.append(acc_val)
        if len(acc_val_list) > 1 and acc_val_list[-1] > acc_val_list[-2]:
            break

def gen_test(test):
    model.eval()
    predictions = []
    encoded_data_test = tokenizer.batch_encode_plus(test.text.values,
                                                    add_special_tokens=True,
                                                    return_attention_mask=True,
                                                    pad_to_max_length=True,
                                                    max_length=256,
                                                    return_tensors='pt'
                                                    )
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(test.label.values)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    dataloader_test = DataLoader(dataset_test, sampler = SequentialSampler(dataset_test), batch_size= 32)

    for batch in dataloader_test:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }
        with torch.no_grad():        
            outputs = model(**inputs)

        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)

    predictions = np.concatenate(predictions, axis=0)
    pre = []
    for result in predictions:
        pre.append(np.argmax(result) - 1)
    return pre


def main():
    raw_train = open("train.txt", 'r', encoding='UTF-8')
    raw_test = open("test.txt", 'r', encoding='UTF-8')
    train, test = data_clean(raw_train, raw_test)
    X_train, X_val, _, _ = train_test_split(train.id.values, train.label.values, test_size=0.15, random_state=17, stratify=train.label.values)
    train['data_type'] = ['not_set']*train.shape[0]   # creat a new column
    train.loc[X_train, 'data_type'] ='train' # set data type to train
    train.loc[X_val, 'data_type'] = 'val' # set data type to val

    # use BERT-Base, Uncased Model which has 12 layers, 768 hidden, 12 heads, 110M parameters
    # train_Bert(train)
    # test_pre = gen_test(test)
    # np.savetxt("pre.txt", np.array(test_pre), delimiter=',', fmt='%1d')

if __name__ == "__main__":
    main()