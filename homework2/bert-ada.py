import os
import random
import numpy as np
import re
import xml.etree.ElementTree as ET
import urllib 
import urllib2 
import requests
import tarfile
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer)
from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)

labels = ['negative', 'neatral', 'positive']

def data_raw2formed(data_dir, train = True):
    def __indent(elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                __indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    label_list = {"-1": "negative", "0": "neutral", "1": "positive"}
    if train == True:
        file = open("train.txt", encoding='utf-8')
        inter = 3
    else:
        file = open("test.txt", encoding='utf-8')
        inter = 2
    lines = file.readlines()
    
    root = ET.Element('sentences')
    tree = ET.ElementTree(root)
    __indent(root)
    for i in range(0, len(lines), inter):
        key = lines[i][:-1]
        term = lines[i+1][:-1]
        label = lines[i+2][:-1]
        pattern = r"\$T\$"
        pattern = re.compile(pattern)
        sentence = re.sub(pattern, term, key)
        sentence = re.sub(u"&",u"and",sentence)
        sentence = re.sub(u"\"",u"",sentence)
        term = re.sub(u"\"",u"", term)
        former = """<sentence id=\"{}\">
        <text>{}</text>
    <aspectTerms>
        <aspectTerm from=\"0\" polarity=\"{}\" term=\"{}\" to=\"0\" />
        </aspectTerms>
    </sentence>""".format(int(i/inter), sentence, label_list[label], term)
        try:
            sent = ET.fromstring(former)
        except:
            print("skip",former)
        root.append(sent)
        __indent(root)
    
    if train == True:
        tree.write(os.path.join(data_dir, "train.xml"), encoding='utf-8', xml_declaration=True)
    else:
        tree.write(os.path.join(data_dir, "test.xml"), encoding='utf-8', xml_declaration=True)
    file.close()


def train(args, train_data, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    
    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs))
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch):
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, train_loss / global_step


def term_to_aspect(filename):
    sentimap = {
        'negative': 'NEG',
        'neutral':  'NEU',
        'positive': 'POS'
    }
    with open(filename) as file:
        sentence_elements = ET.parse(file).getroot().iter('sentence')
        sentences = []
        aspect_term_sentiments = []

        for j, s in enumerate(sentence_elements):
            sentence_text = s.find('text').text
            aspect_term_element = []
            for o in s.iter('aspectTerm'):
                aspect_term = o.get('term')
                classes.add(aspect_term)
                sentiment = sentimap[o.get('polarity')]
                aspect_term_sentiment.append((aspect_term, sentiment))

            if len(aspect_term_sentiment) > 0:
                aspect_term_sentiments.append(aspect_term_sentiment)
                sentences.append(sentence_text)

    return sentences, aspect_term_sentiments
    

def generate_sent_pairs(sentences, aspect_term_sentiments):
    sentence_pairs = []
    labels = []

    for ix, ats in enumerate(aspecterm_sentiments):
        s = sentences[ix]
        for k, v in ats:
            sentence_pairs.append((s, k))
            labels.append(v)

    return sentence_pairs, labels


def create_samples(corpus, set_type):
    sentences, aspects = term_to_aspect(corpus)
    sentences, labels = generate_sent_pairs(sentences, aspects)
    examples = []

    for i, sentence_pair in enumerate(sentences):
        guid = "%s-%s" % (set_type, i)
        try:
            text_a = sentence_pair[0]
            text_b = sentence_pair[1]
            label = labels[i]
        except IndexError:
            print("indexerror")
            continue
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def get_samples(args, pre=False):
    if pre == False:
        return create_samples(os.path.join(args.data_dir, "train.xml"), "train")
    else:
        return create_samples(os.path.join(args.data_dir, "test.xml"), "test")

def sample2Features(samples, args.max_seq_length, tokenizer):


def load_data(args, tokenizer, pre=False):
    def transform_examples_to_hr(exmpls):
        examples_hr = ['[CLS] ' + exp.text_a + ' [SEP] ' + exp.text_b + ' [LABEL] ' + exp.label for exp in exmpls]
        return examples_hr

    logger.info("Creating features from dataset file at %s", args.data_dir)
    samples = get_samples(args.data_dir) if pre == False else get_samples(args.data_dir, pre=True)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    return dataset, transform_examples_to_hr(samples) # tokenized_examples, examples, all_label_ids,   # transform_examples_to_hr(examples)

def pre_test(args, test_data, model, tokenizer):
    test_data, _ = load_data(args, tokenizer, pre=True)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
    preds = []
    for batch in tqdm(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}
            outputs = model(**inputs)
            _, logits = outputs[1]
        
        preds.append(logits.detach().cpu().numpy()):
    preds = np.argmax(preds, axis=1)
    np.savetxt("pre.txt", preds, fmt='%1d')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    args.data_dir = 'data/transformed'
    data_raw2formed(args.data_dir)
    args.model_name_or_path = 'data/models/restaurants_10mio_ep3'
    url = 'https://drive.google.com/drive/folders/175DsRrPpF9df9EBfjmRL4KWG-1uCAM7D/restaurants_10mio_ep3.tar'
    urllib.urlretrieve(url, "restaurants_10mio_ep3.tar")
    filename = 'data/models/restaurants_10mio_ep3.tar'
    tar = tarfile.open(filename, mode = "r:gz")
    tar.extractall()
    tar.close()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained(args.model_name_or_path, num_labels = 3)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = True)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    model.to(args.device)
    train_data, _ = load_data(args, tokenizer, evaluate=False)
    global_step, train_loss = train(args, train_data, model, tokenizer)

    pre_test(args, test_data, model, tokenizer)

if __name__ == "__main__":
    main()