import csv
import numpy as np
from sklearn import svm
import sklearn

def feature_look_down_1(input_pts, input_index):
    output_pts = []
    output_pt = []
    for each_r in input_pts:
        for index in input_index:
            output_pt.append(each_r[index])
        output_pts.append(output_pt)
        output_pt = []
    # print(len(output_pts), len(output_pts[0]))
    return output_pts


# start of data processing part
with open('my_pics/json_datas/ans_all_15_images.csv', mode = 'r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    ans_data = []
    for each in data_reader:
        ans_data = each
    ans_data = list(map(int, ans_data))
with open('my_pics/json_datas/input_all_15_images.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    all_pts = []
    for each in data_reader:
        # print(each)
        all_pts.append(each)
# print('output is ', all_pts) # all points is a 2D array with 15 rows and 75 columns
# this modifies the list into a 15 rows and 50 columns 2D array that has only x and y coordinates
for each in all_pts:
    for pts in each:
        if pts <= 1:
            each.remove(pts)
# data I need is neck (1), Relbow(3), Rwrist(4), Lelbow(6), Lwrist(7), Reye(15), Leye(16)
need_features = [1, 3, 4, 6, 7, 15, 16]
need_idx = []
for each in need_features:
    need_idx.append(each * 2)
    need_idx.append(each * 2 + 1)
# print(need_pts) # this gets the index needed for the research
feature_1 = feature_look_down_1(all_pts, need_idx)
print(len(feature_1))
print(len(ans_data))

# start of sklearn part
X = np.array(feature_1)
Y = np.array(ans_data)

acc = 0
f1_score = 0
score = 0
score2 = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)
    clf = svm.SVC(degree=3, gamma='scale', kernel='rbf', decision_function_shape='ovo')
    # clf = svm.LinearSVC(random_state=0, tol=1e-5)
    # clf2 = sklearn.MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(x_train, y_train)
    # clf2.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # y_pred2 = clf2.predict(x_test)
    # score2 += MLPClassifier.score(clf2, X, Y)
    score += svm.SVC.score(clf, X, Y)
    acc = acc + sklearn.metrics.accuracy_score(y_test, y_pred)
    # print(sklearn.metrics.accuracy_score(y_test, y_pred))
    # f1_score = f1_score + sklearn.metrics.f1_score(y_test, y_pred, average='binary')
# print(f1_score / 1000)
print(acc / 1000)
print(score / 10)
# print(score2 / 1000)



