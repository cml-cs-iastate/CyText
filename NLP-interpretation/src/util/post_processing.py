import os
import pickle as pkl

from operator import itemgetter

import src.settings as settings


# load the importance values of the interpretation_features
with open(os.path.join(settings.output_directory, 'IF.pkl'), 'rb') as f:
    IF = pkl.load(f)

# load the count of the interpretation_features
with open(os.path.join(settings.output_directory, 'count_IF.pkl'), 'rb') as f:
    Freq = pkl.load(f)

# load the average important value of the interpretation_features
with open(os.path.join(settings.output_directory, 'average_IF.pkl'), 'rb') as f:
    Ravg = pkl.load(f)


# final dictionary
final_IF = dict()

mu_threshold = 0.5
beta_threshold = 25
phi_threshold = 0.25

for category, sub_dict in Ravg.items():
    for word, value in sub_dict.items():
        if value >= mu_threshold and Freq[category][word] >= beta_threshold and len(word) >= 2:
            is_true = True
            for category_idx in range(len(Ravg)):
                if category_idx in Ravg and category_idx != category:
                    if word not in Ravg[category_idx] or value >= Ravg[category_idx][word] + phi_threshold:
                        continue
                    else:
                        is_true = False
                        break
            if is_true:
                final_IF.setdefault(category, dict())
                final_IF[category][word] = value

print(f'the length of the interpretation_features dict for all categories: {len(final_IF)}')

# write the sets of true features to files.
with open(os.path.join(settings.output_directory, 'final_IF.pkl'), 'wb') as f:
    pkl.dump(final_IF, f, pkl.HIGHEST_PROTOCOL)


for key, value in final_IF.items():
    print(f'The interpretation features of class: {key} is :')
    for item in sorted(value.items(), key=itemgetter(1)):
        print('{} : {:.2f}'.format(item[0], item[1]))
    print(f'The length of the dictionary for class {key} is = {len(value)}')
    print()
