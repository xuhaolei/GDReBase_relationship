import argparse
import sys

parser = argparse.ArgumentParser()
# 以后再统一吧
parser.add_argument("--dataset",type=str,default='NCBI-disease-IOBES',help='disease or bacterium')
# parser.add_argument("--dataset",type=str,default='ncbi_disease',help='disease or bacterium')
args = parser.parse_known_args() # 提前解析
dataset = args[0].dataset

parser.add_argument("--datatype", type=str, default="disease",help='disease or bacterium')
if dataset == 'ncbi_disease':
	parser.add_argument("--train_path", type=str, default="ncbi_data/NCBItrainset_corpus.txt",help='train_path')
	parser.add_argument("--dev_path", type=str, default="ncbi_data/NCBIdevelopset_corpus.txt",help='dev_path')
	parser.add_argument("--test_path", type=str, default="ncbi_data/NCBItestset_corpus.txt",help='test_path')
	parser.add_argument('--target_size',type=int, default=9)
else:
	parser.add_argument("--train_path", type=str, default="data/%s/train.tsv"%dataset,help='train_path')
	parser.add_argument("--dev_path", type=str, default="data/%s/devel.tsv"%dataset,help='dev_path')
	parser.add_argument("--test_path", type=str, default="data/%s/test.tsv"%dataset,help='test_path')
	parser.add_argument('--target_size',type=int, default=5)

parser.add_argument("--vocab_path", type=str, default="vocab.pkl",help='vocab_path')
parser.add_argument("--tag_vocab_path", type=str, default="tag_vocab.pkl",help='vocab_path')
parser.add_argument("--char_vocab_path", type=str, default="char_vocab.pkl",help='vocab_path')
parser.add_argument("--testonly", action="store_const", const=True, default=False)

# model
parser.add_argument("--hidden_size", type=int, default=300)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--embed_file", type=str, default='word_embed/processed.embed')
parser.add_argument("--embed_size", type=int, default=100)
parser.add_argument("--tag_embed_size", type=int, default=20)
parser.add_argument("--char_embed_size", type=int, default=20)
parser.add_argument("--char_conv_size",type = int,default=30)


parser.add_argument("--tag_vocab_size", type=int, default=38)
parser.add_argument("--char_vocab_size", type=int, default=40)
parser.add_argument("--threadhold", type=float, default=0.1)
parser.add_argument('--ffn_size',type=int, default=100)
parser.add_argument("--test_model", type=str, default=None)

# training
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--pad_size", type=int, default=512)
parser.add_argument("--char_pad_size", type=int, default=20,help='max char in a word')
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--gpu", type=int, default=-1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("--save_path", type=str, default='saved_dict/BiLstm-CRF-%s.pth'%dataset, help='path for save model')
parser.add_argument('--loss_function',default='BCE',type=str,help='BCE or weighted BCE',choices = ['BCE','WBCE'])

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command