import csv
import numpy as np
from sklearn import svm
import sklearn

# start of data processing part
with open('my_/json_datas/ans_all_15_images.csv', mode = 'r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    ans_data = []
    for each in data_reader:
        ans_data = each
    ans_data = list(map(int, ans_data))