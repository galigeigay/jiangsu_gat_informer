# from copy import deepcopy
from datetime import datetime, timedelta

from typing import List, Dict, Any, Optional

# from astral.sun import sun
from astral import LocationInfo
import numpy as np
import pandas as pd

from api.utils.data_info_config import load_data_info_config


data_infos = load_data_info_config('configs/Data_Info.yaml')


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
IDX_FORMAT = "%Y%m%d%H"



def update_necessary_info_to_chart(pdf: pd.DataFrame, info_items: bool = True, update_time: bool = True):
    """
    Update necessary info to mongo chart
    Args:
        pdf: pandas dataframe
        info_items: if add info_date and info_time
        update_time: if add update time

    Returns:

    """
    # Convert date time format to string type
    if pdf["date_time"].dtype == 'datetime64[ns]':
        pdf["date_time"] = pdf["date_time"].dt.strftime(DATE_FORMAT)
    elif pdf["date_time"].dtype != 'object':
        pdf["date_time"] = pdf["date_time"].astype(str)

    if pdf["version"].dtype == 'datetime64[ns]':
        pdf["version"] = pdf["version"].dt.strftime(DATE_FORMAT)
    elif pdf["version"].dtype != 'object':
        pdf["version"] = pdf["version"].astype(str)

    if info_items:
        pdf["info_date"] = pdf["date_time"].str.split(" ").str[0]
        pdf["info_time"] = pd.to_datetime(pdf["date_time"], format=DATE_FORMAT) + timedelta(minutes=15)
        pdf["info_time"] = pdf["info_time"].dt.strftime("%H:%M")

    if update_time:
        pdf['update_time'] = datetime.now().strftime(DATE_FORMAT)
    return pdf


def update_necessary_info_to_doris(pdf: pd.DataFrame, update_list: List[str]):
    for update_item in update_list:
        if update_item == "version":
            if pdf[update_item].dtype == 'datetime64[ns]':
                pdf[update_item] = pdf[update_item].dt.strftime(DATE_FORMAT)
            elif pdf[update_item].dtype != 'object':
                pdf[update_item] = pdf[update_item].astype(str)
            pdf["version_date"] = pdf[update_item].str[:10]
            pdf["version_time"] = pdf[update_item].str[11:]
            pdf.rename(columns={update_item: "data_version"}, inplace=True)
        elif update_item == "date_time":
            if pdf[update_item].dtype == 'datetime64[ns]':
                pdf[update_item] = pdf[update_item].dt.strftime(DATE_FORMAT)
            elif pdf[update_item].dtype != 'object':
                pdf[update_item] = pdf[update_item].astype(str)
            pdf[update_item] = pdf[update_item].astype(str)
            pdf["info_date"] = pdf[update_item].str[:10]
            pdf["info_time"] = pd.to_datetime(pdf["date_time"], format=DATE_FORMAT) + timedelta(minutes=15)
            pdf["info_time"] = pdf["info_time"].dt.strftime("%H:%M")
            pdf["info_time"] = pdf["info_time"].str.replace('00:00', '24:00')
        elif update_item == "version_diff":
            pdf[update_item] = pdf[update_item].astype(int)
            pdf.rename(columns={update_item: "version_diff"}, inplace=True)
        elif update_item == "update_time":
            pdf['update_time'] = datetime.now().strftime(DATE_FORMAT)
        elif update_item == "version_diff":
            pdf['date_time'] = pd.to_datetime(pdf['date_time'])
            pdf['data_version'] = pd.to_datetime(pdf['data_version'])
            pdf['date_time_utc'] = pd.to_datetime(pdf['date_time']) - pd.Timedelta(hours=8)
            pdf['version_diff'] = pdf['date_time_utc'] - pdf['data_version']
            pdf['version_diff'] = ((pdf['version_diff'].dt.total_seconds() / 3600 - 1) // 12).astype(int)
            pdf.drop(columns=['date_time_utc'], inplace=True)
            pdf['date_time'] = pdf['date_time'].dt.strftime(DATE_FORMAT)
            pdf['data_version'] = pdf['data_version'].dt.strftime(DATE_FORMAT)
        else:
            renamed_label_item = update_item if 'ahead' in update_item else "realtime_" + update_item
            pdf.rename(columns={renamed_label_item: "value"}, inplace=True)

    return pdf
            

def group_upload_data(pdf: pd.DataFrame, model_time: str, label_item: str, upload_format: str = "default", 
                      version_name: str = "version_num"):

    if upload_format == "default":
        pdf = update_necessary_info_to_chart(pdf)
        pdf['model_version'] = model_time
        upload_name_list = ["date_time", "version", "source", label_item, "eval_mode", 
                            "model_version", "info_date", "info_time", "update_time", "interpolation"]
        cols_to_drop = [col for col in pdf.columns if col not in upload_name_list]
        pdf = pdf.drop(columns=cols_to_drop)
    elif upload_format == "doris_fmt1":
        proc_list = ["version", "date_time", "update_time", label_item] + [version_name]
        pdf = update_necessary_info_to_doris(pdf, proc_list)
        pdf["node_name"] = "unify"
        pdf["model_name"] = "xgboost"
        pdf["model_version"] = "V3.0"
        pdf['model_log_name'] = model_time

    return pdf

