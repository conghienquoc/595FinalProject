import argparse
import os
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from roco_utils import Model, get_captions, test, load_mlm_data, ROCO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def eval_bleu(model):
#     model.eval()
#     val_captions_fname = os.path.join(args.captions_dir, "captions_val2014.json")
#     val_captions = get_captions(val_captions_fname)

#     val_path = os.path.join(args.data_dir, 'val2014')

#     bleu1 = bleu2 = bleu3 = bleu4 = 0
#     with torch.no_grad():
#         for fname in os.listdir(val_path):
#             id = int(fname.split('.')[0].split('_')[-1])
#             reference = val_captions[id].split()
#             candidate = model()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Eval MMBERT")

    # parser.add_argument('-r', '--run_name', type=str, help="name for wandb run", required=True)
    parser.add_argument('--data_dir', type=str, default = '../../data', help='path to dataset', required = False)
    parser.add_argument('--captions_dir', type=str, default = '../../data/annotations', help='path to captions', required = False)
    parser.add_argument('--model_dir', type=str, default = 'Weights/coco_mlm/val_loss_3.pt', help='path to trained model', required = False)
    parser.add_argument('--mlm_prob', type=float, default=0.15, required = False, help='probability of token being masked')


    parser.add_argument('--num_test_samples', type=int, default=10, help='Test sample size')
    parser.add_argument('--num_val_samples', type=int, default=40000, help='Validation sample size')
    parser.add_argument('--train_pct', type=float, default=1.0, help='fraction of train set')
    parser.add_argument('--valid_pct', type=float, default=1.0, help='fraction of validation set')
    parser.add_argument('--test_pct', type=float, default=1.0, help='fraction of test set ')

    parser.add_argument('--batch_size', type=int, default=16, help='batch_size.')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='rlp patience')
    parser.add_argument('--factor', type=float, default=0.1, help='rlp factor')
    parser.add_argument('--num_workers', type=int, default=4, help='num works to generate data.')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to train')
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--max_position_embeddings', type=int, default=75, help='embedding size')
    parser.add_argument('--n_layers', type=int, default=4, help='num of heads in multihead attenion')
    parser.add_argument('--heads', type=int, default=12, help='num of bertlayers')
    parser.add_argument('--type_vocab_size', type=int, default=2, help='types of tokens ')
    parser.add_argument('--vocab_size', type=int, default=30522, help='vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=768, help='embedding size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3, help='dropout')

    args = parser.parse_args()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model = Model(args)
    model.load_state_dict(torch.load(args.model_dir))
    model.to(device)
    
    train_data, val_data, keywords  = load_mlm_data(args)
    word2idx =  {w:idx for idx,w in enumerate(keywords)}
    idx2word = {idx:w for idx,w in enumerate(keywords)}
    
    num_classes = len(word2idx)

    args.num_classes = num_classes
    train_data = [(fname, get_indexed_caption(caption, word2idx)) for fname, caption in train_data]
    val_data = [(fname, get_indexed_caption(caption, word2idx)) for fname, caption in val_data]
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)


    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
    
    val_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valdataset = ROCO(args, val_data, val_tfm, keywords, mode = 'validation')
    valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    PREDS, total_acc, bleu = test(valloader, model, device, scaler, idx2word)
    
    print("Tested on {} samples\nAccuracy: {}\nBLEU score: {}".format(len(PREDS), total_acc, bleu))



