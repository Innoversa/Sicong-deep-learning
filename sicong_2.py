import csv
import numpy as np
from sklearn.svm import SVC


def feature1(input1):
    output = [0, 1, 2, 3]
    # output[0] = (input1[0][0] - input1[3][0])
    output[1] = (input1[0][1] - input1[3][1])
    # output[2] = (input1[6][0] - input1[3][0])
    output[3] = (input1[6][1] - input1[3][1])
    # output_val = output[0] + output[1] * 10 + output[2] * 100 + output[3] * 1000
    # return [output[1]-output[0], output[3]-output[2]]
    return output


def feature2(input1):
    output = [0, 1, 2, 3, 4, 5, 6]
    output[0] = input1[0][1]
    output[1] = input1[1][1]
    output[2] = input1[2][1]
    output[3] = input1[3][1]
    output[4] = input1[4][1]
    output[5] = input1[5][1]
    output[6] = input1[6][1]
    return output


with open('data.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    outp = []
    for each in data_reader:
        outp.append(each)
    # print(outp[0:7])

with open('answer.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, quoting=csv.QUOTE_NONNUMERIC)
    ans = []
    for each in data_reader:
        ans = each
    ans = list(map(int, ans))
    # print(ans)

f1 = []
for a in range(17):
    f1.append(feature1(outp[a*7:a*7+7]))
    # f1.append(feature2(outp[a * 7:a * 7 + 7]))
print(f1)
print(ans)
X = np.array(f1[3:])
print(X)
Y = np.array(ans[3:])
print(Y)
clf = SVC(gamma='auto')
clf.fit(X, Y)
# print(clf.predict([[-0.8, -1, -0.8, -1]]))
print(clf.predict(f1[0:3]))

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

 