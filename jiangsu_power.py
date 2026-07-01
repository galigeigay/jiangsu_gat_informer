import pandas as pd
import os
import sys
from datetime import datetime,timedelta
from copy import deepcopy

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

from api.io.logger import setup_logger
from api.model.data import get_data
from api.model.xgbmodel import XGBModel
from api.model.pipeline import ModelPipeline,run_model
from api.model.utils import calculate_accuracy
from jobs import daily_energy_job, set_global_timestamp

logger = setup_logger(__name__)


if __name__ == "__main__":

    local_dir = f'{script_path}/local/checkpoints'
    os.makedirs(local_dir, exist_ok=True)

    cur_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    past_two_days = (datetime.now() + timedelta(days=-15)).strftime('%Y-%m-%d-%H%M%S')
    # past_days = (datetime.now() + timedelta(days=-367)).strftime('%Y-%m-%d-%H%M%S')
    past_days = (datetime.now() + timedelta(days=-160)).strftime('%Y-%m-%d-%H%M%S')

    set_global_timestamp(cur_timestamp)
    run_mode = sys.argv[1]
    # 模型类型可在此切换: 'xgboost' | 'gat_informer'
    model_type = 'gat_informer'

    if run_mode == 'train':

        # 实时供需-含日前
        daily_energy_job(
            model_type=model_type,
            train=True,
            label_list=['new_energy_wind_ahead'],
            # label_list=['new_energy_wind_ahead', 'new_energy_solar_ahead', ],
            feature_list = [],
            trade_type='ahead',
            start_date=past_days[:10],
            end_date=past_two_days[:10],
            local_dir=local_dir,
            load_strictly=False,
            get_label=True,
            submit_result=False,
            meteo_scope='city',
            version_num=0,
            meteo_source='HRES',
            capa_rev=True,
            meteo_weighted=False
        )
    elif run_mode == 'predict':

        daily_energy_job(
            model_type=model_type,
            train=False,
            label_list=['new_energy_wind_ahead'],
            # label_list=['new_energy_wind_ahead', 'new_energy_solar_ahead',],
            feature_list = [],
            trade_type='ahead',
            start_span=-12,
            end_span=-1,
            local_dir=local_dir,
            load_strictly=False,
            get_label=True,##True,False
            submit_result=True,
            meteo_scope='city',
            version_num=0,
            meteo_source='HRES',
            capa_rev=True,
            meteo_weighted=False,
            use_interpolation=True
        )


