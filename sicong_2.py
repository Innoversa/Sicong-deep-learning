import csv
import numpy as np
from sklearn.svm import SVC

with open('data.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'',quoting=csv.QUOTE_NONNUMERIC)
    outp = []
    for each in data_reader:
        outp.append(each)
    result = []


# print(Y)
# X = np.array(feature1)
# print(X)
# Y = np.array([1,1,0,1,1,1])
# clf = SVC(gamma='auto')
# clf.fit(X, Y)
# print(clf.predict([[-0.8, -1]]))
print(outp)


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

 