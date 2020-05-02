import csv
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
import sklearn
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split


def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(x_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(x_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=Y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        # print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()



def featureAll(input1):
    print(input1)
    # input("hello")
    output = []
    output += (feature1(input1))
    # visual_trend += (feature2(input1))
    output += (feature3(input1))
    output += feature4(input1[7::])
    return output


def feature1(input1):
    # feature regarding lips
    output = [0, 1, 2, 3]
    output[0] = (input1[0][0] - input1[3][0])
    output[1] = (input1[0][1] - input1[3][1])
    output[2] = (input1[6][0] - input1[3][0])
    output[3] = (input1[6][1] - input1[3][1])
    # output_val = visual_trend[0] + visual_trend[1] * 10 + visual_trend[2] * 100 + visual_trend[3] * 1000
    return [output[1]-output[0], output[3]-output[2]]
    # return visual_trend


def feature2(input1):
    # feature regarding lips
    output = [0, 1, 2, 3, 4, 5, 6]
    output[0] = input1[0][1]
    output[1] = input1[1][1]
    output[2] = input1[2][1]
    output[3] = input1[3][1]
    output[4] = input1[4][1]
    output[5] = input1[5][1]
    output[6] = input1[6][1]
    return output


def feature3(input1):
    # feature regarding lips
    output = [0, 1, 2, 3, 4, 5]
    output[0] = input1[0][0]
    output[1] = input1[0][1]
    output[2] = input1[3][0]
    output[3] = input1[3][1]
    output[4] = input1[6][0]
    output[5] = input1[6][1]
    return output


def feature4(input2):
    output = []
    for e in range(len(input2)):
        output.append(input2[e][0])
    return output


with open('data.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    outp = []
    for each in data_reader:
        outp.append(each)

with open('answer.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, quoting=csv.QUOTE_NONNUMERIC)
    ans = []
    for each in data_reader:
        ans = each
    ans = list(map(int, ans))

with open('fake_smile.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, quoting=csv.QUOTE_NONNUMERIC)
    fake = []
    for each in data_reader:
        fake = each
    fake = list(map(int, fake))

with open('happy_smile.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, quoting=csv.QUOTE_NONNUMERIC)
    happy = []
    for each in data_reader:
        happy = each
    happy = list(map(int, happy))
    # print(ans)

f1 = []
for a in range(17):
    f1.append(featureAll(outp[a * 16:a * 16 + 16]))
    # f1.append(featureAll(outp))
    # f1.append(feature3(outp))
print('printing', f1)
print('len = ', len(f1))
# print(ans)
X = np.array(f1)
# print(X)
Y = np.array(ans)
# print(Y)
acc = 0
f1_score = 0
score = 0
score2 = 0
for i in range(1000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y)
    # clf = svm.SVC(degree=3, gamma='scale', kernel='poly', decision_function_shape='ovr')
    clf = svm.LinearSVC(random_state=0, tol=1e-5)
    # clf2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
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



# # Plot calibration curve for Gaussian Naive Bayes
# plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)
#
# # Plot calibration curve for Linear SVC
# plot_calibration_curve(LinearSVC(max_iter=10000), "SVC", 2)
#
# plt.show()

# print(clf.predict([[-0.8, -1, -0.8, -1]]))
# print(clf.predict(f1))

'''
from sklearn import svm
clf = svm.SVC
clf.fit(pic_fea_list, Label)
predict=clf.predict(test_fea_list)
'''

# # train and test
# print("train and test")
# clf = svm.SVC()
# pic_fea_list_test =[]

# root_test='assets/videoframetest/test'
# for i in range(1,71):
#   path_test = root_test+str(i)+'.jpg'
#   #print('test picture ', i)
#   fea_test = get_feature(path_test)
#   pic_fea_list_test.append(fea_test)
  
#   x = np.array(pic_fea_list)
#   y = np.array(Label)
#   x = x.reshape(x.shape[1:])
#   x = x.transpose()
#   #nsamples, nx, ny = x.shape
#   #d2_train_dataset = x.reshape((nsamples,nx*ny))
  
#   clf.fit(x, y)

 