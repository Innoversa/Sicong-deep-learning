import face_alignment
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import numpy as np

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)


'''
from sklearn import svm
clf = svm.SVC
clf.fit(pic_fea_list, Label)
predict=clf.predict(test_fea_list)
'''
#cross_validation
#f1/f-measure. Evaluation. sklearn has function that you can call

def get_feature(path):
  #try:
  input_img = io.imread(path) #aflw-test
  #except FileNotFoundError:
      #input_img = io.imread(path)
  
  preds = fa.get_landmarks(input_img)[-1]
  
  
  '''
  
  # 2D-Plot
  plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)
  '''
  pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
  pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                }
  '''
  fig = plt.figure(figsize=plt.figaspect(.5))
  ax = fig.add_subplot(1, 2, 1)
  ax.imshow(input_img)
  
  for pred_type in pred_types.values():
      ax.plot(preds[pred_type.slice, 0],
              preds[pred_type.slice, 1],
              color=pred_type.color, **plot_style)
  
  ax.axis('off')
  
  # 3D-Plot
  ax = fig.add_subplot(1, 2, 2, projection='3d')
  surf = ax.scatter(preds[:, 0] * 1.2,
                    preds[:, 1],
                    preds[:, 2],
                    c='cyan',
                    alpha=1.0,
                    edgecolor='b')
  
  for pred_type in pred_types.values():
      ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                preds[pred_type.slice, 1],
                preds[pred_type.slice, 2], color='blue')
  '''
  lips_list=[]
  for item in preds[pred_types['lips'].slice]:
    #lips_list.append(item)  #3D
    lips_list.append(item[:2])  #2D
    
  #print lips_list
  return lips_list
  '''
  ax.view_init(elev=90., azim=90.)
  ax.set_xlim(ax.get_xlim()[::-1])
  plt.show()
  '''
  
  
pic_fea_list=[]
root='/home/wangshuaifang/face-alignment/test/assets/videoframe/pic'
for i in range(1,5):
  path = root+str(i)+'.jpg'
  #print('picture ', i)
  fea = get_feature(path)
  pic_fea_list.append(fea)
  #print pic_fea_list  

Label=[]
file1 = open("assets/label.txt","r")
a=file1.readlines()[0]
file1.close()
a=a.split(',')
for i in a:
  Label.append(int(i))

# print Label

# train and test
print("train and test")
clf = svm.SVC()
pic_fea_list_test =[]

root_test='assets/videoframetest/test'
for i in range(1,71):
  path_test = root_test+str(i)+'.jpg'
  #print('test picture ', i)
  fea_test = get_feature(path_test)
  pic_fea_list_test.append(fea_test)
  
  x = np.array(pic_fea_list)
  y = np.array(Label)
  x = x.reshape(x.shape[1:])
  x = x.transpose()
  #nsamples, nx, ny = x.shape
  #d2_train_dataset = x.reshape((nsamples,nx*ny))
  
  clf.fit(x, y)

 