# For task 1 and 3 I reused the same network "SentimentClassifier". It uses an embedding layer
# therefore allowing to load just the indices of words from the dataset and not a one hot encoding.
# As shown in the output.txt, the network learns the training data pretty well. It does overfit
# but that can be remedied using regularization, batch balancing ..etc.
# I used a max sequence length of 60 which was the mode of the dataset.
# For task 2. I wrappend the TEXTNetOrder2 and used the same embedding layer
# only to find that training was ungodly slow so only was able to report results
# on task 1 and 3.
import torch
from torch import nn
import sys
import os
import gzip
import pickle
import random
import argparse
import numpy as np
from collections import defaultdict
from pprint import pprint as pp

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)
EPOCHS = 6
VALIDATION_BATCH = 5
TRAIN_BATCH = 4

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TEXTnetOrder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TEXTnetOrder2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.combined_to_hidden = nn.Linear(input_dim + 2 * hidden_dim,
                                            hidden_dim)
        self.combined_to_middle = nn.Linear(input_dim + 2 * hidden_dim, 100)
        self.middle_to_out = nn.Linear(100, output_dim)
        # for the cell
        self.cell = None
        self.linear_for_cell = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden):
        if self.cell is None:
            self.cell = torch.zeros(x.shape[0], self.hidden_dim).to(DEVICE)
        combined = torch.cat((x, hidden, self.cell), 1)
        hidden = self.combined_to_hidden(combined)
        out = self.combined_to_middle(combined)
        out = self.middle_to_out(out)
        self.cell = torch.tanh(self.linear_for_cell(hidden))
        return out, hidden


class TEXTnetOrder2Adapter(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_dim,
                 hidden_dim=1024,
                 embed_dim=1024):
        super(TEXTnetOrder2Adapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = TEXTnetOrder2(embed_dim, hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_dim).to(DEVICE)
        embed = self.embed(x)
        for k in range(embed.shape[1]):
            out, hidden = self.rnn(embed[:, k, :], hidden)
        return out


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_file, max_length=60):
        super(SentimentAnalysisDataset, self).__init__()
        self.max_length = max_length
        root_dir = root
        f = gzip.open(os.path.join(root_dir, dataset_file), 'rb')
        dataset = f.read()
        if sys.version_info[0] == 3:
            self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(
                dataset, encoding='latin1')
        else:
            self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(
                dataset)
        self.categories = sorted(list(self.positive_reviews_train.keys()))
        self.category_sizes_train_pos = {
            category: len(self.positive_reviews_train[category])
            for category in self.categories
        }
        self.category_sizes_train_neg = {
            category: len(self.negative_reviews_train[category])
            for category in self.categories
        }
        self.indexed_dataset = []
        length_stats = defaultdict(int)
        for category in self.positive_reviews_train:
            for review in self.positive_reviews_train[category]:
                self.indexed_dataset.append([review, category, 1])
                length_stats[len(review)] += 1

        for category in self.negative_reviews_train:
            for review in self.negative_reviews_train[category]:
                self.indexed_dataset.append([review, category, 0])
                length_stats[len(review)] += 1
        # pp("Review stats {}".format(length_stats))
        # ascii_histogram(length_stats)
        # self.padding = max(length_stats, key=lambda key: length_stats[key])
        # print("Padding with {} max length".format(self.padding))
        random.shuffle(self.indexed_dataset)

    def vocab_size(self):
        return len(self.vocab)

    def one_hotvec_for_word(self, word):
        word_index = self.vocab.index(word)
        hotvec = torch.zeros(1, len(self.vocab))
        hotvec[0, word_index] = 1
        return hotvec

    def review_to_tensor(self, review):
        review_tensor = torch.zeros(self.max_length)
        for i, word in zip(range(self.max_length), review):
            review_tensor[i] = self.vocab.index(word) if i < len(review) else 0
        return review_tensor

    def sentiment_to_tensor(self, sentiment):
        sentiment_tensor = torch.zeros(2)
        if sentiment is 1:
            sentiment_tensor[1] = 1
        elif sentiment is 0:
            sentiment_tensor[0] = 1
        sentiment_tensor = sentiment_tensor.type(torch.long)
        return sentiment_tensor

    def __len__(self):
        return len(self.indexed_dataset)

    def __getitem__(self, idx):
        sample = self.indexed_dataset[idx]
        review = sample[0]
        review_category = sample[1]
        review_sentiment = sample[2]
        review_sentiment = torch.tensor([review_sentiment]).long().squeeze(0)
        review_tensor = self.review_to_tensor(review).long()
        category_index = self.categories.index(review_category)
        sample = {
            'review': review_tensor,
            'category':
            category_index,  # should be converted to tensor, but not yet used
            'sentiment': review_sentiment
        }
        return sample


def ascii_histogram(seq):
    """A horizontal frequency-table/histogram plot."""
    for k in sorted(seq.keys()):
        print('{0:5d} {1}'.format(k, '+' * seq[k]))


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/nhendy/ece695dl/hw6/DLStudio-1.1.3/Examples/data",
        type=str,
        help="Purdue dataset root directory.")
    parser.add_argument("--epochs",
                        default=EPOCHS,
                        type=int,
                        help="Number of epochs to train")
    parser.add_argument("--debug",
                        action='store_true',
                        default=False,
                        help="Debug mode")
    return parser.parse_args(argv)


def _modes():
    return ['train', 'test']


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


class SentimentClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 output_dim,
                 embedding_dim=1024,
                 hidden_dim=512,
                 n_layers=1,
                 dropout_prob=0.02):
        super(SentimentClassifier, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embed = self.embed(x)
        out, hidden = self.gru(embed)
        return self.fc(out[:, -1])


def train_classifier(net,
                     loaders,
                     epochs,
                     input_label_fn,
                     out_file='output.txt',
                     retain=False):
    net = net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    out_fd = open(out_file, "a+")
    for epoch in range(epochs):
        running_loss = {'train': 0.0, 'test': 0.0}
        running_tps = {'train': 0, 'test': 0}
        running_samples = {'train': 0, 'test': 0}
        for mode in _modes():
            if mode == "test":
                net.eval()
            else:
                net.train()
            for i, data in enumerate(loaders[mode]):
                inputs, labels = input_label_fn(data)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_tps[mode] += (outputs.argmax(
                    dim=1) == labels).sum().item()
                running_samples[mode] += inputs.shape[0]
                running_loss[mode] += loss.item()
                if mode == "train":
                    loss.backward(retain_graph=retain)
                    optimizer.step()
                if mode == "train" and i % 100 == 99:
                    print("Training loss, iter {}: {}".format(
                        i + 1, running_loss['train'] / (i + 1)))
                    print(
                        "[epoch {:02d}: iter {:05d}]    loss: {:.03f}".format(
                            epoch + 1, i + 1, running_loss['train'] / (i + 1)),
                        file=out_fd)
        print(
            "Epoch {}: Train Loss {}, Validation Loss {}. Train Accuracy {}, Validation Accuracy {}"
            .format(epoch, running_loss['train'] / len(loaders['train']),
                    running_loss['test'] / len(loaders['test']),
                    running_tps['train'] / running_samples['train'],
                    running_tps['test'] / running_samples['test']))
    compute_confusion_matrix(net, loaders['test'].dataset,
                             ['negative', 'positive'], out_fd)
    out_fd.close()


def print_conf_matrix(class_names, matrix, file_handle=sys.stdout):
    row_format = "{:>10}" + "{:>10.3f}" * (len(class_names))
    head_format = "{:>10}" * (len(class_names) + 1)
    print(head_format.format("", *class_names), file=file_handle)
    for class_name, row in zip(class_names, matrix):
        print(row_format.format(class_name, *row), file=file_handle)


def compute_confusion_matrix(net,
                             test_dataset,
                             class_names,
                             out_file=sys.stdout):
    loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=1,
                                         drop_last=True)
    confusion_matrix = torch.zeros(size=(len(class_names), len(class_names)))
    samples_per_class = torch.zeros(size=(len(class_names), ))
    for i, data in enumerate(loader):
        x, y = data['review'], data['sentiment']
        x = x.to(DEVICE)
        y_pred = torch.argmax(net(x), dim=-1, keepdims=False)
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        confusion_matrix[y, y_pred_cpu] += 1
        samples_per_class[y] += 1
    confusion_matrix /= samples_per_class
    confusion_matrix *= 100
    print_conf_matrix(class_names=class_names,
                      matrix=confusion_matrix.numpy().tolist(),
                      file_handle=out_file)


def train_sentiment_classifier(data_root, epochs, debug):
    loaders = {}
    for mode in _modes():
        dataset = SentimentAnalysisDataset(
            root=data_root,
            dataset_file="sentiment_dataset_{}_{}.tar.gz".format(
                mode, 40 if debug else 200))
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=_batch(mode),
                                                    drop_last=True)
    net = SentimentClassifier(vocab_size=loaders['train'].dataset.vocab_size(),
                              output_dim=2)
    with open('output.txt', "a+") as fd:
        print("Task 1: ", file=fd)
    train_classifier(net, loaders, epochs, lambda data:
                     (data['review'], data['sentiment']))


def train_textnet_classifier(data_root, epochs, debug):
    loaders = {}
    for mode in _modes():
        dataset = SentimentAnalysisDataset(
            root=data_root,
            dataset_file="sentiment_dataset_{}_{}.tar.gz".format(
                mode, 40 if debug else 200),
            max_length=10)
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=1,
                                                    drop_last=True)
    net = TEXTnetOrder2Adapter(
        vocab_size=loaders['train'].dataset.vocab_size(), output_dim=2)
    train_classifier(net,
                     loaders,
                     epochs,
                     lambda data: (data['review'], data['sentiment']),
                     '/tmp/test.txt',
                     retain=False)


def task_one(args):
    with open('output.txt', "a+") as fd:
        print("Task 1: ", file=fd)
    train_sentiment_classifier(args.root, args.epochs, args.debug)


def task_two(args):
    train_textnet_classifier(args.root, args.epochs, args.debug)


def task_three(args):
    with open('output.txt', "a+") as fd:
        print("Task 3: ", file=fd)
    train_sentiment_classifier(args.root, args.epochs, args.debug)


def main(argv):
    args = parse_args(argv)
    task_three(args)


if __name__ == "__main__":
    main(sys.argv[1:])
