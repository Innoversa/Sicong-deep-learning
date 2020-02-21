import csv
import numpy
import sklearn
# import sicong_2 as sicong_func
from sklearn.model_selection import train_test_split
from sklearn import svm


def read_files():
    with open('data.csv', mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        out_t = []
        for each in data_reader:
            out_t.append(each)
    with open('fake_smile.csv', mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        out_f = []
        for each in data_reader:
            out_f = each
        out_f = list(map(int, out_f))
    with open('happy_smile.csv', mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        out_h = []
        for each in data_reader:
            out_h = each
        out_h = list(map(int, out_h))
    return out_t, out_f, out_h  # returning 3 lists (X, Y1, and Y2)
# 17 x 16. 6 from lip, 5 from eyebrow1, 5 from eyebrow2


def feature_1_raised_eyebrow(input1):
    output1 = [0, 0]
    output1[0] = abs((input1[6][1]-input1[10][1]) / (input1[6][0]-input1[10][0]))
    output1[1] = abs((input1[11][1] - input1[15][1]) / (input1[11][0] - input1[15][0]))
    return input1   # outputting two index as one feature


def feature_2_dimpler(input1):
    output1 = [0, 0, 0, 0]
    output1[0] = input1[0][0]
    output1[1] = input1[5][0]
    output1[2] = input1[2][0]
    output1[3] = input1[2][1]
    return output1  # outputting four index as one feature


def feature_all(input1):
    output = []
    output += feature_1_raised_eyebrow(input1)
    output += feature_2_dimpler(input1)
    # output += sicong_func.featureAll(input1)
    return output


data_arr, ans_arr1, ans_arr2 = read_files()
f1 = []
for a in range(17):
    f1.append(feature_all(data_arr[a * 16:a * 16 + 16]))

X = numpy.array(f1)
Y1 = numpy.array(ans_arr1)
Y2 = numpy.array(ans_arr2)
print(f1)
print(ans_arr1)
ans1_score = 0
ans2_score = 0
input("hello")
for i in range(100):
    x_train1, x_test1, y_train1, y_test1 = sklearn.model_selection.train_test_split(X, Y1)
    x_train2, x_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(X, Y2)
    clf1 = sklearn.svm.SVC(gamma='auto', kernel='rbf', decision_function_shape='ovo')
    clf2 = sklearn.svm.SVC(gamma='auto', kernel='rbf', decision_function_shape='ovo')
    clf1.fit(x_train1, y_train1)
    clf2.fit(x_train2, y_train2)
    y_pred1 = clf1.predict(x_test1)
    y_pred2 = clf2.predict(x_test2)
    ans1_score += sklearn.svm.SVC.score(clf1, X, Y1)
    ans2_score += sklearn.svm.SVC.score(clf2, X, Y2)
print('ans1 == '+ans1_score)
print('ans2 == '+ans2_score)