import csv

file_path = 'C:\\Users\\moham\\Documents\\Projects\\Text_Classification\\data\\Yelp\\Yelp.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    incsv = csv.reader(f)
    header = next(incsv)  # Header
    label_idx = header.index('label')
    content_idx = header.index('content')

    max_len, avg_len, total_words_count, number_of_samples, pos_labels, neg_labels = 0, 0, 0, 0, 0, 0
    vocab = set()

    for line in incsv:
        # count number of samples
        number_of_samples += 1

        # count the negative and positive samples
        if line[label_idx] == '0':
            neg_labels += 1
        elif line[label_idx] == '1':
            pos_labels += 1

        count = 0
        for word in line[content_idx].split():
            if word not in vocab:
                vocab.add(word)
            count += 1

        total_words_count += count

        # get the maximum length sample
        if count > max_len:
            max_len = count

    # get the average length
    avg_len = total_words_count / number_of_samples

    # get the size of the vocabulary
    vocab_size = len(vocab)
    print(f'maximum length: {max_len}')
    print(f'average length: {avg_len}')
    print(f'total number of words: {total_words_count}')
    print(f'number of samples: {number_of_samples}')
    print(f'number positive_labels: {pos_labels}')
    print(f'number negative_labels: {neg_labels}')
    print(f'vocab size: {vocab_size}')
