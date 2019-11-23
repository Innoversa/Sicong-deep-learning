import face_alignment
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import csv

def get_img_data(path):
    # Run the 3D face alignment on a test image, without CUDA.
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)


    input_img = io.imread(path)


    preds = fa.get_landmarks(input_img)[-1]

    # 2D-Plot
    plot_style = dict(marker='o',
                    markersize=4,
                    linestyle='-',
                    lw=2)

    pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
    # pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
    #             'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
    #             'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
    #             'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
    #             'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
    #             'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
    #             'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
    #             'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
    #             'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
    #             }
    pred_types = {'lips': pred_type(slice(54, 61), (0.596, 0.875, 0.541, 0.3))}
    # print(preds['lips'].slice)
    # lips = slice(52, 60)
    data = preds[pred_types['lips'].slice, 0:2]

    # fig = plt.figure(figsize=plt.figaspect(.5))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(input_img)
    #
    #
    # for pred_type in pred_types.values():
    #     ax.plot(preds[pred_type.slice, 0],
    #             preds[pred_type.slice, 1],
    #             color=pred_type.color, **plot_style)
    #
    # ax.axis('off')

    # 3D-Plot

    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax.scatter(preds[:, 0] * 1.2,
    #                 preds[:, 1],
    #                 preds[:, 2],
    #                 c='cyan',
    #                 alpha=1.0,
    #                 edgecolor='b')
    #
    # for pred_type in pred_types.values():
    #     ax.plot3D(preds[pred_type.slice, 0] * 1.2,
    #             preds[pred_type.slice, 1],
    #             preds[pred_type.slice, 2], color='blue')
    #
    # ax.view_init(elev=90., azim=90.)
    # ax.set_xlim(ax.get_xlim()[::-1])
    # plt.show()
    # plt.savefig("sample.jpg")
    return data


# path = '../test/assets/aflw-test.jpg'
# path = 'pics/dimpler_1.jpg'
# data = get_img_data(path)

with open('data.csv', mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # data_writer.writerow(['x_value', 'y_value'])
    for i in range(13):
        if i < 10:
            path = 'pics/dimpler_0'+str(i)+'.png'
        else:
            path = 'pics/dimpler_'+str(i)+'.png'
        print(path)
        data = get_img_data(path)
        data_writer.writerows(data)



#pic1.jpg pic2.jpg....

# for i in range(100):
#     path = '../test/assets/pic'+str(i)+'.jpg'
#     data = get_img_data(path)
