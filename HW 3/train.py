import argparse
from utils_ic import load_data, read_jason
from model_ic import NN_Classifier, validation, make_NN, save_checkpoint

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")

""" ADDED 3 EXTRA ARGUMENTS TO MAKE make_NN FUNCTION MORE FLEXIBLE (WILBERT ARISTO) """
parser.add_argument("--pretrain", type=bool, default=False, help="pretrained model or not")
parser.add_argument("--finetune_whole", type=bool, default=False, help="finetune whole model or not")
parser.add_argument("--custom_model_with_layer", type=int, default=0 , help="use custom model or not")
""" ============================================================================================== """

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model = make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data,\
                pretrain=args.pretrain, finetune_whole = args.finetune_whole, custom_model=args.custom_model_with_layer)

if args.save_dir:
    save_checkpoint(model, args.save_dir)