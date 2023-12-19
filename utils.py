from collections import defaultdict
import numpy as np
from time import time
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import heapq
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,\
                        classification_report, precision_recall_fscore_support

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

def calculate_performance(datasets, preds, labels, masks, preds_softmax, losses):

    avg_loss    = round(np.sum(losses)/np.sum(masks),4)
    avg_acc     = [round(accuracy_score(labels, preds, sample_weight=masks)*100,2) if datasets=='iemocap' 

            else round(f1_score(labels, preds, sample_weight=masks, average='macro')*100,2)  if datasets=='dailydialog'
            else round(f1_score(labels, preds, sample_weight=masks, average='micro')*100,2)]
                    
    avg_fscore  = [round(f1_score(labels,preds,sample_weight=masks,average='weighted')*100,2)] \
        if datasets != 'dailydialog' else [round(f1_score(labels,preds, sample_weight=masks, average='micro', labels=[0,2,3,4,5,6])*100, 2)]
    avg_list    = [labels, preds, masks]

    return avg_loss, avg_acc,  avg_fscore,  avg_list

def final_print(best_loss, best_label, best_pred, best_mask):
    print('Test performance..')

    print(classification_report(best_label,best_pred,sample_weight=best_mask,digits=4))
    np.set_printoptions(suppress=True)
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))    

def compute_loss(loss_list, weights=[1.0, 0.0]):

    return sum(loss*weight for loss, weight in zip(loss_list, weights))

def tSNE_plot(embeds, label, save_path, dim=2):
    tsne = TSNE(n_components=dim, init='pca', random_state=0)
    t0 = time()
    data = tsne.fit_transform(embeds)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                color=plt.cm.Set1(label[i] / 10.),
                fontdict={'weight': 'bold', 'size': 3})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE embedding of the digits (time %.2fs)'% (time() - t0))
    fig.savefig(save_path + '_pic-best.png',dpi=300)
    return 

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.cont_lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs_cont)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class logging_results:
    def __init__(self, key_res, type_res=list) -> None:
        self.key_res = key_res
        self.len_res = len(key_res)
        self.type_res= type_res
        self.res_dict = defaultdict(list)

    def logging(self, ret_value):
        assert len(ret_value) == self.len_res
        for i, k in enumerate(self.key_res):
            self.res_dict[k].append(ret_value[i])

    def get_result(self, keys):
        assert [k in self.key_res for k in keys]
        return [self.res_dict[k] for k in keys]

    def get_results(self):
        if self.type_res == list:
            return [np.concatenate(v) for v in self.res_dict.values()]
        elif self.type_res == int:
            return [v[-1] for v in self.res_dict.values()]
        else:
            raise NotImplementedError("Only List and Int is supported.")