import csv

dataset_file = 'Amazon 250 Reviews Preprocessed Nov 6.csv'
with open(dataset_file, 'r', encoding='utf-8', errors='ignore') as f:
    print('Building dataset ...')
    csv_reader = csv.reader(f)
    next(csv_reader)
    data, A, B = list(), list(), list()
    for sample in csv_reader:
        data.append(sample[0])
        A.append(sample[3])
        B.append(sample[4])

    # compare between A and B (Human with Human / or Human with Machine)
    a, b, c, d, counter = 0, 0, 0, 0, 0
    for idx, sample in enumerate(data):
        list_A = A[idx].split()
        list_B = B[idx].split()
        for word in sample.split():
            counter += 1
            if word in list_A and word in list_B:
                a += 1
            elif word in list_A and word not in list_B:
                b += 1
            elif word not in list_A and word in list_B:
                c += 1
            elif word not in list_A and word not in list_B:
                d += 1

    po = (a + d) / (a + b + c + d)
    p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
    p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
    pe = p1 + p0
    k = (po - pe) / (1 - pe)

    print(f'a: {a}, b: {b}, c: {c}, d: {d}, sum(a,b,c,d): {a+b+c+d}, len(data): {counter}')
    print(f'po: {po}, p0: {p0}, p1: {p1}, pe: {pe}, Kappa: {k}')
