"""
University of Amsterdam: MSc AI: NLP2 2019 Spring
Project 2: Sentence VAE
Team 3: Gururaja P Rao, Manasa J Bhat
"""

import matplotlib.pyplot as plt


def plot(data, epoch, type, title, legend, save_path):
    if type == 'loss':
        y_label = "NLL Loss -->"
    else:
        y_label = "Perplexity -->"

    x_values = [i for i in range(epoch + 1)]
    for i, y_values in enumerate(data):
        plt.plot(x_values, y_values, label=legend[i])
    plt.xlabel("Epochs -->")
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()