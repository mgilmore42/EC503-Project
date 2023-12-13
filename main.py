import argparse

def download_callback():
    import utils
    utils.download_all()

def train_callback():
    import algos.train as train
    train.train_all()

parser = argparse.ArgumentParser(description='Process some integers.')

subparsers = parser.add_subparsers(help='sub-command help')

# Create the download subparser
download_parser = subparsers.add_parser('download', help='Download data')
download_parser.set_defaults(func=download_callback)  # Set the callback function for the download subcommand

# Create the train subparser
train_parser = subparsers.add_parser('train', help='Train a model')
train_parser.set_defaults(func=train_callback)  # Set the callback function for the train subcommand

# Parse the command-line arguments
args = parser.parse_args()

# Access the arguments
if hasattr(args, 'func'):
    func_args = vars(args).copy()
    del func_args['func']
    args.func(**func_args)
else:
    parser.print_help()
