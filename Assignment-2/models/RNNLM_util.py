"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import os
import sys
import numpy as np
import torch
from torch import nn
import time


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(detach_hidden(h) for h in hidden)


def train_model(model, dataset, epoch, lr, opt):
    print("-----------------------------------Training-----------------------------------")
    model.train()
    total_loss = []
    hidden = model.init_hidden(opt.batch_size)
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.train_data)
    start = time.time()
    perplexity = []

    for batch, idx in enumerate(range(0, dataset.train_data.size(0) - 1, opt.seq_length)):
        source, target = dataset.load_data('train', idx)
        hidden = detach_hidden(hidden)
        model.zero_grad()
        output, hidden = model(source, hidden)
        loss = criterion_loss(output.view(-1, vocab_size), target)
        loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss.append(loss.item())
        ppl = np.exp(loss.item()) / opt.seq_length
        perplexity.append(ppl)

        if (batch % opt.print_interval == 0) and batch != 0:
            elapsed_time = (time.time() - start) * 1000 / opt.print_interval
            print('Epoch: {:5d} | {:5d}/{:5d} batches | LR: {:5.4f} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
                epoch, batch, data_size // opt.seq_length, lr, np.mean(total_loss), np.mean(perplexity), elapsed_time))
            start = time.time()


    print('\nEpoch: {:5d} | Average loss: {:5.4f} | Average Perplexity : {:5.4f}'.format(
        epoch, np.mean(total_loss), np.mean(perplexity)))

    return np.mean(total_loss), np.mean(perplexity)


def validate_model(model, dataset, epoch, opt):
    print("----------------------------------Validation----------------------------------")
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(opt.test_batch)
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.val_data)
    start = time.time()
    perplexity = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.val_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('val', idx)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()

            perplexity += np.exp(loss.item()) * len(source)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, elapsed_time))

    return loss, per_word_ppl


def test_model(model, dataset, epoch, opt):
    print("----------------------------------Testing----------------------------------")
    model.eval()
    total_loss = 0.0
    hidden = model.init_hidden(opt.test_batch)
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.test_data)
    start = time.time()
    perplexity = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.test_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('test', idx)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(-1, vocab_size)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()

            perplexity += np.exp(loss.item()) * len(source)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, elapsed_time))

    return loss, per_word_ppl


def generate_sentences(model, dataset, sentence_len):
    print("Generating sentence using the trained model\n\n")
    model.eval()
    hidden = model.init_hidden(1)
    vocab_size = len(dataset.vocabulary)
    input = torch.randint(vocab_size, (1, 1), dtype=torch.long)

    sentence = []
    with torch.no_grad():
        for i in range(sentence_len):
            output, hidden = model(input, hidden)
            # Do multinomial sampling and pick the word with max weight
            word_weights = output.squeeze().exp()
            word_idx = torch.multinomial(word_weights, 1)[0]
            # Add the new word to the input sequence
            input.fill_(word_idx)
            word = dataset.vocabulary.vocab[word_idx]
            sentence.append(word)

    final_sentence = '\t'
    for word in sentence:
        if word == '-SOS-':
            final_sentence = final_sentence + '\t'
        elif word == '-EOS-':
            final_sentence = final_sentence + ' .\n'
        else:
            final_sentence = final_sentence + ' ' + word

    print(final_sentence)