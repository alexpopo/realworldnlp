import sys
sys.path.append("/content/realworldnlp")

from itertools import chain
from allennlp.training import GradientDescentTrainer
from allennlp.data.data_loaders import MultiProcessDataLoader

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader

from examples.sentiment.sst_classifier import LstmClassifier
from realworldnlp.predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    # In order to use ELMo, each word in a sentence needs to be indexed with
    # an array of character IDs.
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(token_indexers={'tokens': elmo_token_indexer})

    train_path = 'data/stanfordSentimentTreebank/trees/train.txt'
    dev_path = 'data/stanfordSentimentTreebank/trees/dev.txt'

    # Initialize the ELMo-based token embedder using a pre-trained file.
    # This takes a while if you run this script for the first time

    # Original
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # Medium
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

    # Use the 'Small' pre-trained model
    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    train_dataset = reader.read(train_path)
    dev_dataset = reader.read(dev_path)

    # You can optionally specify the minimum count of tokens/labels.
    # `min_count={'tokens':3}` here means that any tokens that appear less than three times
    # will be ignored and not included in the vocabulary.
    # vocab = Vocabulary.from_instances(chain(train_data_loader.iter_instances(), dev_data_loader.iter_instances()))
    vocab = Vocabulary.from_instances(chain(train_dataset, dev_dataset))

    sampler = BucketBatchSampler(batch_size=32, sorting_keys=["tokens"])
    train_data_loader = MultiProcessDataLoader(reader, train_path, batch_sampler=sampler)
    dev_data_loader = MultiProcessDataLoader(reader, dev_path, batch_sampler=sampler)

    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

    # Pass in the ElmoTokenEmbedder instance instead
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    # The dimension of the ELMo embedding will be 2 x [size of LSTM hidden states]
    elmo_embedding_dim = 256
    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, lstm, vocab)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    trainer = GradientDescentTrainer(
        model=model,
        optimizer=optimizer,
        data_loader=train_data_loader,
        validation_data_loader=dev_data_loader,
        patience=2,  # 10
        num_epochs=5,  # 20
        cuda_device=-1)

    trainer.train()

    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict('This is the best movie ever!')['logits']
    label_id = np.argmax(logits)
    print()
    print(model.vocab.get_token_from_index(label_id, 'labels'))


if __name__ == '__main__':
    main()
