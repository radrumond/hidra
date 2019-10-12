## Created by Rafael Rego Drumond and Lukas Brinkmeyer
# THIS IMPLEMENTATION USES THE CODE FROM: https://github.com/dragen1860/MAML-TensorFlow

from data_gen.omni_gen import unison_shuffled_copies,OmniChar_Gen, MiniImgNet_Gen
from archs.fcn   import Model as mfcn
from archs.hydra import Model as mhyd
from train import *
from test  import *
from args import argument_parser, train_kwargs, test_kwargs
import random

args = argument_parser().parse_args()
random.seed(args.seed)
t_args = train_kwargs(args)
e_args = test_kwargs (args)

print("########## argument sheet ########################################")
for arg in vars(args):
    print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
print("##################################################################")

print("Loading Data...")
if args.dataset in ["Omniglot", "omniglot", "Omni", "omni"]:
    loader = OmniChar_Gen  (args.data_path)
    isMIN = False
    shaper = [28,28,1]
elif args.dataset in ["miniimagenet", "MiniImageNet", "mini"]:
    loader = MiniImgNet_Gen(args.data_path)
    isMIN = True
    shaper = [84,84,3]
else:
    raise ValueError("INVALID DATA-SET NAME!")

print("Building Model...")
if   args.arch == "fcn"or args.arch == "maml":
    print("SELECTED: MAML")
    m  = mfcn (meta_lr = args.meta_step, train_lr = args.learning_rate, image_shape=shaper, isMIN=isMIN, label_size=args.max_classes)
    mt = mfcn (meta_lr = args.meta_step, train_lr = args.learning_rate, image_shape=shaper, isMIN=isMIN, label_size=args.max_classes)
#elif args.arch == "rnn":
#    m = mrnn (meta_lr = args.meta_step, train_lr = args.learning_rate, image_shape=shaper, isMIN=isMIN, label_size=args.min_classes)
elif args.arch == "hydra" or args.arch == "hidra":
    print("SELECTED: HIDRA")
    m  = mhyd (meta_lr = args.meta_step, train_lr = args.learning_rate, image_shape=shaper, isMIN=isMIN, label_size=args.max_classes)
    mt = mhyd (meta_lr = args.meta_step, train_lr = args.learning_rate, image_shape=shaper, isMIN=isMIN, label_size=args.max_classes)
else:
    raise ValueError("INVALID Architecture NAME!")

mode = "train"
if args.test:
    mode = "test"
    print("Starting Test Step...")
    mt.build  (K = args.test_inner_K, meta_batchsz = args.meta_batch, mode=mode)
    test (mt, loader, **e_args)
else:
    modeltest = None
    if args.testintrain:
        mt.build (K = args.test_inner_K, meta_batchsz = args.meta_batch, mode="test")
        modeltest = mt
    print("Starting Train Step...")
    m.build  (K = args.train_inner_K, meta_batchsz = args.meta_batch, mode=mode)
    train(m, modeltest, loader, **t_args)
