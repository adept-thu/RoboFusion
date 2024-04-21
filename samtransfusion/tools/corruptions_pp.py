import os
import json
import pandas as pd
import argparse

def json_parse(json_p,model_n, corruption, severity, csv_f):
    '''
    new_df = pd.DataFrame(columns=["model_n", "corruption", "severity", "mAP", "NDS", "car", "truck","bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"])
    '''
    # json_p = r'/sda/dxg/bev/3D_Corruptions_AD/bevfusion/output/metrics_summary.json'
    

    with open(json_p, 'rb') as f:
        metrics_summary = json.load(f)
        
    mAP = metrics_summary["mean_ap"]
    mAP = round(mAP,4)
    NDS = metrics_summary["nd_score"]
    NDS = round(NDS,4)
    car_AP = metrics_summary["mean_dist_aps"]["car"]
    car_AP = round(car_AP,3)
    truck_AP = metrics_summary["mean_dist_aps"]["truck"]
    truck_AP = round(truck_AP,3)
    bus_AP = metrics_summary["mean_dist_aps"]["bus"]
    bus_AP = round(bus_AP,3)
    trailer_AP = metrics_summary["mean_dist_aps"]["trailer"]
    trailer_AP = round(trailer_AP,3)
    construction_vehicle_AP = metrics_summary["mean_dist_aps"]["construction_vehicle"]
    construction_vehicle_AP = round(construction_vehicle_AP,3)
    pedestrian_AP = metrics_summary["mean_dist_aps"]["pedestrian"]
    pedestrian_AP = round(pedestrian_AP,3)
    motorcycle_AP = metrics_summary["mean_dist_aps"]["motorcycle"]
    motorcycle_AP = round(motorcycle_AP,3)
    bicycle_AP = metrics_summary["mean_dist_aps"]["bicycle"]
    bicycle_AP = round(bicycle_AP,3)
    traffic_cone_AP = metrics_summary["mean_dist_aps"]["traffic_cone"]
    traffic_cone_AP = round(traffic_cone_AP,3)
    barrier_AP = metrics_summary["mean_dist_aps"]["barrier"]
    barrier_AP = round(barrier_AP,3)

    csv_f.loc[len(csv_f.index)] = [model_n] + [corruption] + [severity] + [mAP] + [NDS] + [car_AP] + [truck_AP] + [bus_AP] + [trailer_AP] + [construction_vehicle_AP] + [pedestrian_AP] + [motorcycle_AP] + [bicycle_AP] + [traffic_cone_AP] + [barrier_AP]

def main(args):
    # 读取该目录下的json metric日志
    path = args.path
    dataset = args.dataset
    model_n = args.model
    new_df = pd.DataFrame(columns=["model_n", "corruption", "severity", "mAP", "NDS", "car", "truck","bus", "trailer", "construction_vehicle", "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"])
    # output/transfusion_nusc_voxel_LC_cor
    sub_path = r'transfusion_nusc_voxel_LC_cor'
    # sub_path = r'nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser'
    json_path_list = os.path.join(path, sub_path)
    for json_f in sorted(os.listdir(json_path_list)):
        json_f_abs = os.path.join(json_path_list,json_f)
        if os.path.isfile(json_f_abs):
            print(json_f.split('.')[0])
            corruption_severity = json_f.split('.')[0]
            corruption, severity = corruption_severity[:-2], corruption_severity[-1:]
            json_parse(json_f_abs, model_n, corruption, severity, new_df)
    
    out_file = os.path.join(path, f'{model_n}_{dataset}.csv')
    new_df.to_csv(out_file, index=False)
    print(f'save as {out_file}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--path', type=str, default='./output')
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument('--model', type=str, default='bevfusion-det')
    args = parser.parse_args()

    main(args)
# end main