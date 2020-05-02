import json
import os
import csv

def func_process_json_input(digit):
    digit = str(digit)
    path_to_json = 'video_dir_2\\data_src\\may_'+digit+'\\json\\'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    # print(json_files)
    ans_of_it = []
    for each in json_files:
        with open(path_to_json+each) as json_file:
            data = json.load(json_file)
            # print(json.dumps(data, indent=4))
            # print(data['people'][0]['pose_keypoints_2d']) # extracts useful json data
            ans_of_it.append(data['people'][0]['pose_keypoints_2d'])

    print(ans_of_it)
    print(len(ans_of_it), len(ans_of_it[0]))

    data_writer = csv.writer(open('video_dir_2\\data_src\\may_'+digit+'\\may_'+digit+'_pts.csv', 'w+', newline=""),
                             delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # for each in ans_of_it:
    #     print(each)
    data_writer.writerows(ans_of_it)
for each in range(20):
    func_process_json_input(each+1)