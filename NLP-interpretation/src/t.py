import pickle as pkl
from operator import itemgetter

# with open('Output/Amazon/interpretation_features_dicts/average_IF.pkl', 'rb') as f, open('Output/Amazon/interpretation_features_dicts/count_IF.pkl', 'rb') as g:
#     if_average = pkl.load(f, encoding='bytes')
#     if_count = pkl.load(g, encoding='bytes')
#
#     post_processed_if_dict = post_processing(if_average, if_count, alpha_threshold=0.1, beta_threshold=0.15, print_if_dict=False)
#
#     for key, value in post_processed_if_dict.items():
#         print(f'The interpretation features of class: {key} with length {len(value)} :')
#         for item in sorted(value.items(), key=itemgetter(1)):
#             print('{} : {:.2f}'.format(item[0], item[1]))
#         print()
#
#     with open(settings.post_processed_IF, 'wb') as f:
#         pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)
