import csv
import numpy as np
from sklearn import svm
import sklearn
import json


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

def calc_accuracy(X, Y):
    acc = 0
    f1_score = 0
    score = 0
    score2 = 0
    for i in range(1000):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)
        clf = svm.SVC(gamma='auto', kernel='rbf')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score += svm.SVC.score(clf, X, Y)
        acc = acc + sklearn.metrics.accuracy_score(y_test, y_pred)
    print(acc / 1000)
    print(score / 10)
    return clf


# start of data processing part
def input_data(in_dig):
    pts_var = 'video_dir_1/visual_trend/avi_'+str(in_dig)+'/avi_'+str(in_dig)+'_pts.csv'
    ans_var = 'video_dir_1/visual_trend/avi_'+str(in_dig)+'/avi_'+str(in_dig)+'_ans.csv'
    with open(pts_var, mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        all_pts = []
        for each in data_reader:
            # print(each)
            all_pts.append(each)
    # print(len(all_pts), len(all_pts[0])) all_pts is a 13 by 75 matrix while 13 is the time axis
    with open(ans_var, mode='r') as data_file:
        data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
        ans_data = []
        for each in data_reader:
            ans_data = each
        ans_data = list(map(int, ans_data))
    # print(len(ans_data))
    return all_pts, ans_data

#processing the pts
def do_deep_learning(in_dig):
    all_pts, ans_data = input_data(in_dig)
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
    # print(need_idx)
    feature_1 = feature_look_down_1(all_pts, need_idx)
    # print(len(feature_1))
    # print(len(ans_data))

    # start of sklearn part
    X = np.array(feature_1)
    Y = np.array(ans_data)
    clf = calc_accuracy(X, Y)
    # dec_func_ans = clf.decision_function(X)
    dec_func_ans = clf.predict(X)
    dec_func_ans = dec_func_ans.tolist()
    print(dec_func_ans)
    # for each in dec_func_ans:
    #     if each < 0:
    #         each = 1 + each
    # print (dec_func_ans)
    # initializing json visual_trend for Dr. Jiang's required format
    dict_to_json = {}
    dict_list = []
    for i in range(len(dec_func_ans)):
        dict_list.append([i * 5, dec_func_ans[i]])
    dict_to_json['head down'] = dict_list
    print(dict_to_json)
    out_var = 'video_dir_1/visual_trend/avi_'+str(in_dig)+'/avi_'+str(in_dig)+'_json_out.json'
    with open(out_var, 'w+') as outfile:
        json.dump(dict_to_json, outfile)
    print('\n\n end of ', in_dig)


# do_deep_learning(7)
for i in range(8):
    do_deep_learning(i+1)
