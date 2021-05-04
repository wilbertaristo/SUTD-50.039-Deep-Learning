""" CUSTOM FILE CREATED BY ME TO RUN SAVED MODEL ON TEST DATASET(WILBERT ARISTO) """
import argparse
from utils_ic import load_data, read_jason
from model_ic import load_model, test_model

parser = argparse.ArgumentParser(description="Test image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

model = load_model("/content/gdrive/MyDrive/densenet169_model.pth")

test_model(model, testloader)
""" ============================================================================ """