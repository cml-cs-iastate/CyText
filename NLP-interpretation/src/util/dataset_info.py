import csv

import src.settings as settings

with open('../' + settings.data_source, 'r', encoding='utf-8', errors='ignore') as f:
    incsv = csv.reader(f)
    header = next(incsv)  # Header
    label_idx = header.index('label')
    content_idx = header.index('content')

    max_len, total_words, total_samples = 0, 0, 0
    samples_per_class = dict()
    vocab = set()

    for line in incsv:
        # count number of samples
        total_samples += 1

        # count the number of samples per class
        samples_per_class.setdefault(line[label_idx], 0)
        samples_per_class[line[label_idx]] += 1

        # vocab = set(list(vocab) + line[content_idx].split())

        total_words += len(line[content_idx].split())

        # get the maximum length sample
        max_len = max(max_len, len(line[content_idx].split()))

    # get the average length
    avg_len = total_words / total_samples

    # get the size of the vocabulary
    print('Dataset Information')
    print('-------------------')
    print(f'Maximum sample length: {max_len}')
    print(f'Average sample length: {round(avg_len)}')
    print(f'Total number of words: {total_words}')
    print(f'Total number of samples: {total_samples}')
    print(f'Vocab size: {len(vocab)}')
    for key, value in samples_per_class.items():
        print(f'Total number of samples of class {key} : {value}')
