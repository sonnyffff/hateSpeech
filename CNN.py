import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from argparse import Namespace

from sklearn.metrics import precision_recall_fscore_support, classification_report


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
        """
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Retrieve the index associated with the token

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)  # mask_index is 0
        self.unk_index = self.add_token(self._unk_token)  # unk_index is 1
        self.begin_seq_index = self.add_token(self._begin_seq_token)  # begin_seq_index is 2
        self.end_seq_index = self.add_token(self._end_seq_token)  # end_seq_index is 3

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

class ContentVectorizer(object):
    """ Vectorizer to coordinate the Vocabulary"""

    def __init__(self, content_vocab, label_vocab):
        self.content_vocab = content_vocab
        self.label_vocab = label_vocab

    def vectorize(self, content, vector_length=-1, min_seq_len=4):
        """
        Args:
            content (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized content (numpy.array)
        """
        indices = [self.content_vocab.begin_seq_index]
        indices.extend(self.content_vocab.lookup_token(token)
                       for token in content.split(" "))
        indices.append(self.content_vocab.end_seq_index)

        if len(indices) < min_seq_len:
            pad_needed = min_seq_len - len(indices)
            indices += [self.content_vocab.mask_index] * pad_needed

        # Handle truncation
        if vector_length < 0:
            vector_length = len(indices)
        indices = indices[:vector_length]  # truncate if necessary

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.content_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, hatespeech_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            hatespeech_df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary
        Returns:
            an instance of the ContentVectorizer
        """
        label_vocab = Vocabulary()
        for label in sorted(set(hatespeech_df.label)):
            label_vocab.add_token(label)

        word_counts = Counter()
        for content in hatespeech_df.content:
            for token in content.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        content_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                content_vocab.add_token(word)

        return cls(content_vocab, label_vocab)

class ContentDataset(Dataset):
    def __init__(self, hatespeech_df, vectorizer):
        """
        Args:
            hatespeech_df (pandas.DataFrame): the dataset
            vectorizer (ContentVectorizer): vectorizer instatiated from dataset
        """
        self.hatespeech_df = hatespeech_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, hatespeech_df.content)) + 2

        self.train_df = self.hatespeech_df[self.hatespeech_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.hatespeech_df[self.hatespeech_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.hatespeech_df[self.hatespeech_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # Class weights
        class_counts = hatespeech_df.label.value_counts().to_dict()

        def sort_key(item):
            return self._vectorizer.label_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, hatespeech_csv):
        """Load dataset and make a new vectorizer from scratch
        Args:
            hatespeech_csv (str): location of the dataset
        Returns:
            an instance of ContentDataset
        """
        hatespeech_df = pd.read_csv(hatespeech_csv)
        hatespeech_df = hatespeech_df.rename(columns={'Content': 'content', 'Label': 'label'})

        # hatespeech_df = hatespeech_df.sample(n=5000, random_state=42).reset_index(drop=True)

        train_content_df = hatespeech_df[hatespeech_df.split == 'train']
        return cls(hatespeech_df, ContentVectorizer.from_dataframe(train_content_df))

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        content_vector = \
            self._vectorizer.vectorize(row.content, self._max_seq_length)

        label_index = \
            self._vectorizer.label_vocab.lookup_token(row.label)

        return {'x_data': content_vector,
                'y_target': label_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class ContentClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels,
                 hidden_dim, num_classes, dropout_p,
                 pretrained_embeddings=None, padding_idx=0):
        """
        Args:
            embedding_size (int): size of the embedding vectors
            num_embeddings (int): number of embedding vectors
            filter_width (int): width of the convolutional kernels
            num_channels (int): number of convolutional kernels per layer
            hidden_dim (int): the size of the hidden dimension
            num_classes (int): the number of classes in classification
            dropout_p (float): a dropout parameter
            pretrained_embeddings (numpy.array): previously trained word embeddings
                default is None. If provided,
            padding_idx (int): an index representing a null position
        """
        super(ContentClassifier, self).__init__()

        if pretrained_embeddings is None:
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)

            # in_channels: embedding_size; out_channels: # of filters; kernel_size = n-gram size
        # number of parameters: (# of filters, embedding_size, n-gram size), (100, 100, 2) for 2-gram
        self.conv1d_4gram = nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=4)
        self.conv1d_3gram = nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3)
        self.conv1d_2gram = nn.Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=2)

        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels * 3,
                             hidden_dim)  # input:concatination of conv1d_4gram, conv1d_3gram, conv1d_2gram outputs
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the classifier
        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, dataset._max_seq_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """

        # embed and permute so features are channels
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = F.elu(self.conv1d_4gram(x_embedded))
        remaining_size = features.size(dim=2)
        features_4gram = F.max_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.elu(self.conv1d_3gram(x_embedded))
        remaining_size = features.size(dim=2)
        features_3gram = F.max_pool1d(features, remaining_size).squeeze(dim=2)

        features = F.elu(self.conv1d_2gram(x_embedded))
        remaining_size = features.size(dim=2)
        features_2gram = F.max_pool1d(features, remaining_size).squeeze(dim=2)

        features = torch.cat([features_4gram, features_3gram, features_2gram], dim=1)

        features = F.dropout(features, p=self._dropout_p, training=self.training)

        # mlp classifier
        intermediate_vector = F.dropout(F.relu(self.fc1(features)), p=self._dropout_p, training=self.training)
        prediction_vector = self.fc2(intermediate_vector)  # (batch, num_classes)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])
                train_state['early_stopping_best_val'] = loss_t  # update 'early_stopping_best_val'

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings
    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r", encoding='utf8') as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings

if __name__ == '__main__':
    args = Namespace(
        # Data and Path hyper parameters
        hatespeech_csv="HateBinaryDataset/HateSpeechDatasetBalanced_with_splits.csv",
        model_state_file="cnn_model.pth",
        save_dir="model_storage",
        # Model hyper parameters
        glove_filepath='glove/glove.6B.100d.txt',
        use_glove=True,
        embedding_size=100,
        hidden_dim=100,
        num_channels=100,
        # Training hyper parameter
        seed=1337,
        learning_rate=0.001,
        dropout_p=0.1,
        batch_size=128,
        num_epochs=100,
        early_stopping_criteria=5,
        # Runtime option
        cuda=True,
        catch_keyboard_interrupt=True,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True
    )

    if args.expand_filepaths_to_save_dir:
        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)
        print("Expanded filepaths: ")
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    handle_dirs(args.save_dir)

    args.use_glove = True

    # create dataset and vectorizer
    dataset = ContentDataset.load_dataset_and_make_vectorizer(args.hatespeech_csv)
    vectorizer = dataset.get_vectorizer()

    # Use GloVe to initialized embeddings
    if args.use_glove:
        words = vectorizer.content_vocab._token_to_idx.keys()
        embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                           words=words)
        print("Using pre-trained embeddings")
    else:
        print("Not using pre-trained embeddings")
        embeddings = None

    classifier = ContentClassifier(embedding_size=args.embedding_size,
                                   num_embeddings=len(vectorizer.content_vocab),
                                   num_channels=args.num_channels,
                                   hidden_dim=args.hidden_dim,
                                   num_classes=len(vectorizer.label_vocab),
                                   dropout_p=args.dropout_p,
                                   pretrained_embeddings=embeddings,
                                   padding_idx=0)

    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)

    loss_func = nn.CrossEntropyLoss(dataset.class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=1)

    train_state = make_train_state(args)

    epoch_bar = tqdm(desc='training routine', total=args.num_epochs, position=0)

    dataset.set_split('train')
    train_bar = tqdm(desc='split=train', total=dataset.get_num_batches(args.batch_size), position=1, leave=True)

    dataset.set_split('val')
    val_bar = tqdm(desc='split=val', total=dataset.get_num_batches(args.batch_size), position=1, leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on

            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(batch_dict['x_data'])  # (batch, seq_len) -> (batch, num_classes)

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss, acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = classifier(batch_dict['x_data'])

                # step 3. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'])
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)
                val_bar.set_postfix(loss=running_loss, acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")

    classifier.load_state_dict(torch.load(train_state['model_filename'], weights_only=False))

    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    y_true_list = []
    y_pred_list = []

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(batch_dict['x_data'])

        y_pred_list.extend(y_pred.max(dim=1)[1].cpu().numpy())
        y_true_list.extend(batch_dict['y_target'].cpu().numpy())

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {};".format(train_state['test_loss']))
    print("Test Accuracy: {}".format(train_state['test_acc']))

    print(classification_report(y_true_list, y_pred_list))

