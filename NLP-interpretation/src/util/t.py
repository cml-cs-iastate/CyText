import csv
with open('../data/AmazonYelp.csv', 'r', encoding='utf-8', errors='ignore') as f, open('AmazonYelpNew.txt', 'w', encoding='utf-8') as w:
    csv_reader = csv.reader(f)

    counter = 0
    for line in csv_reader:
        if line[0] == '5':
            counter += 1
            if counter % 5 != 0:
                continue
        if line[0] != '3':
            w.write(line[0]+','+line[1].strip())
            w.write('\n')
