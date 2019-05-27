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

def kl_weight_function(anneal, step, k=0.0025, x0=2500):
    if anneal == 'Logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal == 'Linear':
        return min(1, step / x0)
    else:
        assert False, 'Wrong KL annealing function'

def train_model(model, dataset, epoch, lr, opt):
    print("-----------------------------------Training-----------------------------------")
    model.train()

    # Ignore the padding tokens for the loss computation
    pad_index = dataset.vocabulary.word2token['-PAD-']
    criterion_loss = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.train_data)
    start = time.time()
    total_loss = []
    numerator = 0.0
    denominator = 0.0
    perplexity = []
    accuracy = []

    for batch, idx in enumerate(range(0, data_size - 1, opt.batch_size)):
        source, target, sentence_len = dataset.load_data('train', idx, opt.batch_size)
        if source is None:
            continue

        if torch.cuda.is_available():
            source = source.cuda()
            target = target.cuda()
            hidden = hidden.cuda()

        model.zero_grad()
        embeddings, logv, mean, std = model.encode(source, opt.batch_size)
        output, _ = model.decode(embeddings, mean, std, opt.batch_size, num_samples=opt.sample_size)
        output = output.view(opt.batch_size * opt.seq_length, vocab_size)
        target = target.view(opt.batch_size * opt.seq_length)
        NLL_loss = criterion_loss(output, target)
        # Get the KL loss term and the weight
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_weight_function(anneal=opt.anneal, step=batch)
        loss = (NLL_loss + KL_weight * KL_loss)
        loss.backward(retain_graph=True)
        total_loss.append(loss.cpu().item() / (opt.batch_size * opt.seq_length))

        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        # Compute the perplexity
        numerator += loss.cpu().item()
        denominator += np.sum(sentence_len)
        ppl = np.exp(numerator / denominator)
        perplexity.append(ppl)

        # Compute the word prediction accuracy
        output = output.view(-1, opt.batch_size, vocab_size)
        target = target.view(-1, opt.batch_size)
        acc = compute_accuracy(output, target)
        accuracy.append(acc)

        if (batch % opt.print_interval == 0) and batch != 0:
            elapsed_time = (time.time() - start) * 1000 / opt.print_interval
            print('Epoch: {:5d} | {:5d}/{:5d} batches | LR: {:5.4f} | loss: {:5.4f} | Perplexity : {:5.4f} | Time: {:5.0f} ms'.format(
                epoch, batch, data_size // opt.batch_size, lr, np.mean(total_loss), np.mean(perplexity), elapsed_time))
            start = time.time()
            numerator = 0.0
            denominator = 0.0

    print('\nEpoch: {:5d} | Average loss: {:5.4f} | Average Perplexity : {:5.4f} | Average Accuracy : {:5.4f}'.format(
        epoch, np.mean(total_loss), np.mean(perplexity), np.mean(accuracy)))

    return np.mean(total_loss), np.mean(perplexity), np.mean(accuracy)


def validate_model(model, dataset, epoch, opt):
    print("----------------------------------Validation----------------------------------")
    model.eval()

    total_loss = 0.0
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.val_data)
    start = time.time()
    perplexity = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.val_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('val', idx)
            output, mean, logv, z = model(source, opt.test_batch)
            loss = criterion_loss(output.view(-1, vocab_size), target)
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
    criterion_loss = nn.CrossEntropyLoss()
    vocab_size = len(dataset.vocabulary)
    data_size = len(dataset.test_data)
    start = time.time()
    perplexity = 0.0
    accuracy = 0.0

    with torch.no_grad():
        for batch, idx in enumerate(range(0, dataset.test_data.size(0) - 1, opt.seq_length)):
            source, target = dataset.load_data('test', idx)
            output, mean, logv, z = model(source, opt.test_batch)
            loss = criterion_loss(output.view(-1, vocab_size), target)
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
    vocab_size = len(dataset.vocabulary)
    input = torch.randint(vocab_size, (1, 1), dtype=torch.long)

    sentence = []
    with torch.no_grad():
        for i in range(sentence_len):
            output, _, _, _ = model(input, 1)
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