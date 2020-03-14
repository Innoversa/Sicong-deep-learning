import csv

with open('my_pics/json_datas/input_all_15_images.csv', mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='\'', quoting=csv.QUOTE_NONNUMERIC)
    all_pts = []
    for each in data_reader:
        # print(each)
        all_pts.append(each)
# print('output is ', all_pts) # all points is a 2D array with 15 rows and 75 columns
print(len(all_pts[0]))
