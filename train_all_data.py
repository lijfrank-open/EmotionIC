import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import random
import argparse
import time
import pickle

import torch
import torch.optim as optim


from utils import calculate_performance, final_print, compute_loss, logging_results
from dataloader import get_Emo_loaders, get_train_hyparameters
from model import EmotionIC
from losses import DialogueConLoss

def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=80, metavar='E',
                        help='number of epochs')
    parser.add_argument('--emo',  type=bool, default=True,
                        help='emotion or sentiment')
    parser.add_argument('--dataset',  type=str, default='iemocap',
                        help='the name of dataset, iemocap/meld/emorynlp/dailydialog')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate')
    parser.add_argument('--crf-lr', type=float, default=7e-3, metavar='LR',
                        help='crf learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--hidden-dim', type=int, default=768,
                        help='hidden dim')
    parser.add_argument('--trans-n-layers', type=int, default=5, 
                        help='attention layer number ')
    parser.add_argument('--indi-n-layers', type=int, default=3,
                        help='dialogue GRU layer number')
    parser.add_argument('--use-dropout', action='store_false', default=True,
                        help='use dropout')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--save-path', type=str, default='saved_dict',
                        help='model saved path')
    parser.add_argument('--pretrain-path', type=str, default='saved_dict/best_model_iemocap.pt',
                        help='model load path')
    parser.add_argument('--use-pretrain', action='store_true', default=False,
                        help='pretrain model')
    parser.add_argument('--seed', type=int, default=2023, metavar='seed', help='seed')    

    opt = parser.parse_args()

    opt.cuda = torch.cuda.is_available() and not opt.no_cuda
    if opt.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    return opt


def seed_everything(input_seed):
    global seed
    seed = input_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_or_eval_model(args, model, criterion, dataloader, optimizer=None, epoch=0.1, train=False):

    keys = ['preds', 'labels', 'masks', 'preds_softmax', 'losses']
    log_results = logging_results(keys)

    assert not train or optimizer
    if train:
        model.train()
    else:
        model.eval()    


    features_list = []
    logits_list   = []
    labels_list   = []
    umask_list    = []
    qmask_list    = []


    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        r1, _, _, _, \
        identity_labels, umask, labels = [d.cuda() for d in data[:-1]] if args.cuda else data[:-1]

        dialogues = r1
        labels_dialog = [labels, identity_labels, umask]

        logits, features = model(dialogues, umask, identity_labels)
        loss_list,  pred_list = criterion([logits, 0], labels_dialog)

        features_list.append(features.clone().detach().cpu().numpy())
        logits_list.append(logits.clone().detach().cpu().numpy())
        labels_list.append(labels.clone().detach().cpu().numpy())
        umask_list.append(umask.clone().detach().cpu().numpy())
        qmask_list.append(identity_labels.clone().detach().cpu().numpy())

        loss = compute_loss(loss_list, weights=[1.0, 0.0])

        if train:

            loss.backward() 
            optimizer.step()

        log_results.logging([np.concatenate(pred_list[0]),
                            labels.T.reshape(-1).data.cpu().numpy(),
                            umask.T.reshape(-1).data.cpu().numpy(),
                            torch.argmax(logits,-1).T.reshape(-1).data.cpu().numpy(),
                            loss.item()*umask.view(-1).cpu().numpy()])

    return calculate_performance(args.dataset, *log_results.get_results())


def set_model(args):
    
    args.dropout = args.dropout
    args.attn_drop = args.dropout
    args.feed_drop = args.dropout
    args.rnn_drop = args.dropout

    dialogue_model_hyparam = [args.hidden_dim, args.num_class,
                            args.trans_n_layers, args.indi_n_layers,
                            args.dropout, args.attn_drop, args.feed_drop, args.rnn_drop, args.use_dropout]

    model = EmotionIC(*dialogue_model_hyparam)

    if args.use_pretrain:
        model_dict = model.state_dict()
        checkpoint = torch.load(args.pretrain_path, map_location="cpu")
        state_dict = checkpoint['model']

        if args.dataset not in args.pretrain_path:
            state_dict = {k: v for k, v in state_dict.items() if not 'crf.' in k and 'fc_out.' not in k}
        model_dict.update(state_dict)
        msg = model.load_state_dict(model_dict, strict=False)
        print('The import of the pre-trained model is as follows: {}'.format(msg))

    criterion = DialogueConLoss(num_tags = args.num_class, loss_weights=args.loss_weights)

    if args.cuda:
        model.cuda()
        criterion.cuda()

    base_params = model.parameters()
    optimizer = optim.Adam(
                        [{'params': base_params, 'lr': args.lr},
                        {'params':  criterion.parameters(), 'lr': args.crf_lr},],     
                            weight_decay=args.l2)    

    return model, optimizer, criterion


def main():
    args = parse_option()

    args.loss_weights, args.num_class, _ = get_train_hyparameters(args.dataset)

    print(args)

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(comment="dataset_{}_batch_{}_lr_{}_crf_{}_supmode_{}".format(args.dataset, args.batch_size, args.lr, args.crf_lr, args.sup_mode))

    seed_everything(args.seed)  

    train_loader, valid_loader, test_loader =\
            get_Emo_loaders(datasets_name=args.dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    model, optimizer, criterion = set_model(args)

    keys = ['best_loss', 'best_acc', 'best_fscore', 'best_label', 'best_pred', 'best_mask']
    log_results = logging_results(keys, type_res=int)

    for epoch in range(args.epochs):       
        start_time = time.time()
        train_loss, train_acc, train_fscore, _          = train_or_eval_model(args, model, criterion, train_loader, optimizer, epoch=epoch, train=True)
        valid_loss, valid_acc, valid_fscore, _          = train_or_eval_model(args, model, criterion, valid_loader)
        test_loss,  test_acc,  test_fscore,  test_list  = train_or_eval_model(args, model, criterion, test_loader,epoch=epoch)


        best_fscore = log_results.get_result(['best_fscore'])[0]
        if best_fscore == [] or best_fscore[-1] < test_fscore[0]:
            log_results.logging([test_loss, test_acc, test_fscore[0], *test_list])

        if args.tensorboard:
            writer.add_scalar('test: f1-score/loss', test_fscore[0]/test_loss,    epoch)
            writer.add_scalar('train: f1-score/loss',train_fscore[0]/train_loss,  epoch)
        if args.dataset == 'iemocap':
            logger_prints  = 'epoch:{0:2}|tr_loss:{1:>2.4} tr_acc:{2:>2.4} tr_wef1:{3:>2.4}|v_loss:{4:>2.4} v_acc:{5:>2.4} v_wef1:{6:>2.4}|te_loss:{7:>2.4} te_acc:{8:>2.4} te_wef1:{9:>2.4}|time:{10:>2.3}'
        elif args.dataset == 'dailydialog':
            logger_prints  = 'epoch:{0:2}|tr_loss:{1:>2.4} tr_maf1:{2:>2.4} tr_mif1:{3:>2.4}|v_loss:{4:>2.4} v_maf1:{5:>2.4} v_mif1:{6:>2.4}|te_loss:{7:>2.4} te_maf1:{8:>2.4} te_mif1:{9:>2.4}|time:{10:>2.3}'
        else:
            logger_prints  = 'epoch:{0:2}|tr_loss:{1:>2.4} tr_mif1:{2:>2.4} tr_wef1:{3:>2.4}|v_loss:{4:>2.4} v_mif1:{5:>2.4} v_wef1:{6:>2.4}|te_loss:{7:>2.4} te_mif1:{8:>2.4} te_wef1:{9:>2.4}|time:{10:>2.3}'

        logger_results = [epoch+1, train_loss, train_acc[0], train_fscore[0], valid_loss, valid_acc[0], valid_fscore[0],\
                                    test_loss, test_acc[0], test_fscore[0], round(time.time()-start_time,2)]
        print(logger_prints.format(*logger_results))    

        if (epoch+1)%10 == 0:
            
            final_print(*[v[-1] for v in log_results.get_result(['best_loss', 'best_label', 'best_pred', 'best_mask'])])

            print('-'*150)        

    final_print(*[v[-1] for v in log_results.get_result(['best_loss', 'best_label', 'best_pred', 'best_mask'])])

if __name__ == '__main__':
    main()