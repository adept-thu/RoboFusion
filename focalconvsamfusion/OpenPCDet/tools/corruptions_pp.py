'''
读取指定目录下的txt
保存到同一个csv文件下
'''
import pandas as pd
import os
import argparse
from datetime import datetime

def find_roi_str(line, rois):
    for roi in rois:
        index = line.find(roi)
        if index != -1:
            return index, roi
    return -1, '' # default return

def log_process(model_n, corruption, severity, log_path, csv_f):
    # TODO select window
    log_file_path = log_path
    roi_str = ['Performance of EPOCH', \
         'Car AP_R40@0.70, 0.70, 0.70:', \
         'Pedestrian AP_R40@0.50, 0.50, 0.50:', \
         'Cyclist AP_R40@0.50, 0.50, 0.50:']
    epoch_num_list = []
    result = {}
    date_format = '%Y-%m-%d %H:%M:%S'
    # load txt
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        # calculate timediff
        start_time_str = lines[0][:19]
        end_time_str = lines[-1][:19]
        start_time = datetime.strptime(start_time_str, date_format)
        end_time = datetime.strptime(end_time_str, date_format)
        time_diff = str(end_time - start_time)
        for i, line in enumerate(lines):
            
            line = line.rstrip()
            if len(line)==0: continue

            # mathcing str
            index, roi = find_roi_str(line, roi_str)

            # processing
            if roi == roi_str[0]: 
                cur_epoch = line[index+1+len(roi_str[0]):index+len(roi_str[0])+3]
                epoch_num_list.append(cur_epoch)
                result[f'epoch_{cur_epoch}'] = {} # new epoch
            elif roi in roi_str:
                subdict = {}
                line = lines[i+2].rstrip()
                line.rstrip()
                roi, _ = roi.split(' ', 1)
                item, value = line.split(':', 1)
                item, _ = item.split(' ', 1)
                value_list = value.split(', ')

                subdict[roi.lower()] = {item : value_list}

                line = lines[i+3].rstrip()
                line.rstrip()
                item, value = line.split(':', 1)
                item, _ = item.split(' ', 1)
                value_list = value.split(', ')
                subdict[roi.lower()].update({item: value_list})
                result[f'epoch_{cur_epoch}'].update(subdict) 
    
    # three cls
    for epoch, data in result.items():
        data1 = data['car']['3d'] + data['car']['bev']
        try:
            data1 += data['pedestrian']['3d'] + data['pedestrian']['bev'] 
            data1 += data['cyclist']['3d'] + data['pedestrian']['bev']
        except:
            data1 += ['NaN', 'NaN', 'NaN'] + ['NaN', 'NaN', 'NaN']
            data1 += ['NaN', 'NaN', 'NaN'] + ['NaN', 'NaN', 'NaN']
        
        csv_f.loc[len(csv_f.index)] = [model_n]+ [corruption]+ [severity] + [time_diff] + [epoch] + data1
    # new_df.to_csv(args.output, index=False)


def main(args):
    # 读取该目录下的txt日志
    path = args.path
    dataset = args.dataset
    if dataset == 'kitti':
        dataset = 'kitti_models'
    else:
        raise NotImplementedError
    model = args.model

    new_df = pd.DataFrame(columns=[
                            'model_n', 'corruption_type', 'severity', 'total_time', 'epoch',\
                            'car3d_easy', 'car3d_mod', 'car3d_hard',\
                            'carbev_easy', 'carbev_mod', 'carbev_hard',\
                            'ped3d_easy', 'ped3d_mod', 'ped3d_hard',\
                            'pedbev_easy', 'pedbev_mod', 'pedbev_hard',\
                            'cyc3d_easy', 'cyc3d_mod', 'cyc3d_hard',\
                            'cycbev_easy', 'cycbev_mod', 'cycbev_hard'])

    dir = os.path.join(path, dataset, model) # output/model
    for corruption in sorted(os.listdir(dir)):
        if 'csv' in corruption:
            continue
        dir2 = os.path.join(dir, corruption, 'eval') # output/model/corruption/eval
        for epoch in sorted(os.listdir(dir2)):
            dir3 = os.path.join(dir2, epoch, 'val', 'default') # output/model/corruption/eval/epoch/val/default/
            for log in sorted(os.listdir(dir3)):
                if 'pkl' in log:
                    continue
                print(log)
                model_n = model
                corruption = corruption
                severity = log[log.find(']')+2]
                log_p = os.path.join(dir3, log) # output/model/corruption/eval/epoch/val/default/*.txt
                log_process(model_n, corruption, severity, log_p, new_df)
                
                
    new_df.to_csv(os.path.join(dir, f'{model}_{dataset}.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--path', type=str, default='../output')
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--model', type=str, default='sam_onlinev3')
    args = parser.parse_args()

    main(args)