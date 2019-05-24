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
    pad_index = dataset.vocabulary.word2token['-PAD-']
    criterion_loss = nn.CrossEntropyLoss(ignore_index=pad_index)
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.train_data)
    start = time.time()
    perplexity = []
    accuracy = []

    for batch, idx in enumerate(range(0, data_size, opt.batch_size)):
        source, target, sentence_len = dataset.load_data('train', idx, opt.batch_size)
        hidden = detach_hidden(hidden)
        model.zero_grad()
        output, hidden = model(source, hidden)
        output = output.view(opt.batch_size * opt.seq_length, vocab_size)
        target = target.view(opt.batch_size * opt.seq_length)
        loss = criterion_loss(output, target)
        loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss.append(loss.item())
        ppl = np.exp(loss.item())
        perplexity.append(ppl)

        # Compute the word prediction accuracy
        output = output.view(-1, opt.batch_size, vocab_size)
        target = target.view(-1, opt.batch_size)
        acc = compute_accuracy(output, target)
        accuracy.append(acc)

        if (batch % opt.print_interval == 0) and batch != 0:
            elapsed_time = (time.time() - start) * 1000 / opt.print_interval
            print('Epoch: {:5d} | {:5d}/{:5d} batches | LR: {:5.4f} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
                epoch, batch, data_size // opt.seq_length, lr, np.mean(total_loss), np.mean(perplexity), elapsed_time))
            start = time.time()

        if (batch % 100 == 0) and batch != 0:
            break

    print('\nEpoch: {:5d} | Average loss: {:5.4f} | Average Perplexity : {:5.4f} | Average Accuracy : {:5.4f}'.format(
        epoch, np.mean(total_loss), np.mean(perplexity), np.mean(accuracy)))

    return np.mean(total_loss), np.mean(perplexity), np.mean(accuracy)


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
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, data_size, opt.test_batch)):
            source, target, sentence_len = dataset.load_data('val', idx, opt.test_batch)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(opt.test_batch * opt.seq_length, vocab_size)
            target = target.view(opt.test_batch * opt.seq_length)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()

            # Compute the perplexity
            perplexity += np.exp(loss.item()) * len(source)

            # Compute the word prediction accuracy
            output = output.view(-1, opt.test_batch, vocab_size)
            target = target.view(-1, opt.test_batch)
            accuracy += compute_accuracy(output, target)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    accuracy = accuracy / batch
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Accuracy : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, accuracy, elapsed_time))

    return loss, per_word_ppl, accuracy


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
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, data_size, opt.test_batch)):
            source, target, sentence_len = dataset.load_data('test', idx, opt.test_batch)
            hidden = detach_hidden(hidden)
            output, hidden = model(source, hidden)
            output = output.view(opt.test_batch * opt.seq_length, vocab_size)
            target = target.view(opt.test_batch * opt.seq_length)
            loss = criterion_loss(output, target)
            total_loss += len(source) * loss.item()
            perplexity += np.exp(loss.item()) * len(source)

            output = output.view(-1, opt.test_batch, vocab_size)
            target = target.view(-1, opt.test_batch)
            accuracy += compute_accuracy(output, target)

    loss = total_loss / data_size
    per_word_ppl = perplexity / vocab_size
    accuracy = accuracy / batch
    elapsed_time = (time.time() - start) * 1000 / opt.print_interval
    print('Epoch: {:5d} | loss: {:5.4f} | Perplexity : {:5.4f} | Accuracy : {:5.4f} | Time: {:5.0f} ms'.format(
        epoch, loss, per_word_ppl, accuracy, elapsed_time))


def compute_accuracy(output, target):
    output = torch.argmax(output, dim=2)
    correct = (target == output).float()
    accuracy = torch.mean(correct)

    return accuracy


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
            # Do multinomial sampling and pick the next word
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