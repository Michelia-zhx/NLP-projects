import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import util
from dataset import ExhaustiveDataset, gen_sentence_tensors

def evaluate(model, data_url):
    print("\nEvaluating model use data from ", data_url, "\n")
    max_region = model.max_region
    dataset = ExhaustiveDataset(data_url, next(model.parameters()).device, max_region=max_region)
    data_loader = DataLoader(dataset, batch_size=100, collate_fn=dataset.collate_func)
    # switch to eval mode
    model.eval()

    region_true_list = list()
    region_pred_list = list()
    region_true_count = 0
    region_pred_count = 0

    with torch.no_grad():
        for data, labels, records_list in data_loader:
            batch_region_labels = torch.argmax(model.forward(*data), dim=1).cpu()
            lengths = data[1]
            batch_maxlen = lengths[0]
            for region_labels, length, true_records in zip(batch_region_labels, lengths, records_list):
                pred_records = {}
                ind = 0
                for region_size in range(1, max_region + 1):
                    for start in range(0, batch_maxlen - region_size + 1):
                        end = start + region_size
                        if 0 < region_labels[ind] < dataset.n_tags and end <= length:
                            pred_records[(start, start + region_size)] = region_labels[ind]
                        ind += 1

                region_true_count += len(true_records)
                region_pred_count += len(pred_records)

                for region in true_records:
                    true_label = dataset.categories.index(true_records[region])
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

    print(classification_report(region_true_list, region_pred_list,
                                target_names=dataset.categories, digits=6))

    ret = dict()
    tp = 0
    for pv, tv in zip(region_pred_list, region_true_list):
        if pv == tv:
            tp += 1
    fp = region_pred_count - tp
    fn = region_true_count - tp

    ret['precision'], ret['recall'], ret['f1'] = util.calc_f1(tp, fp, fn)
    return ret

def predict(model, sentences, categories, data_url):
    max_region = model.max_region
    device = next(model.parameters()).device
    tensors = gen_sentence_tensors(
        sentences, device, data_url)
    pred_regions_list = torch.argmax(model.forward(*tensors), dim=1).cpu()

    lengths = tensors[1]
    pred_sentence_records = []
    for pred_regions, length in zip(pred_regions_list, lengths):
        pred_records = {}
        ind = 0
        for region_size in range(1, max_region + 1):
            for start in range(0, lengths[0] - region_size + 1):
                if 0 < pred_regions[ind] < len(categories):
                    pred_records[(start, start + region_size)] = categories[pred_regions[ind]]
                ind += 1
        pred_sentence_records.append(pred_records)
    return pred_sentence_records

def predict_on_iob2(model, iob_url):
    save_url = iob_url.replace('.iob2', '.pred.txt')
    print("predicting on {} \n the result will be saved in {}".format(
        iob_url, save_url))
    test_set = ExhaustiveDataset(iob_url, device=next(
        model.parameters()).device)

    model.eval()
    with open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
        for sentence, records in test_set:
            pred_result = str(predict(model, [sentence], test_set.categories, iob_url)[0]).strip().split(' ')
            label = []
            if len(pred_result) > 1:
                print(pred_result)
                for i in range(0,len(pred_result),3):
                    start = pred_result[i][2:-1]
                    end = pred_result[i+1][0:-2]
                    wordtype = pred_result[i+2][1:-2]
                    label.append("{0},{1} G#{2}".format(start, end, wordtype))
            save_file.write('|'.join(label))