# from datetime import datetime, timedelta
import datetime
from typing import Sequence
import numpy as np
import pandas as pd
import os
from api.model.pipeline import ModelPipeline
from api.model.utils import calculate_accuracy, smooth_gaussian
from api.io.utils import convert_pdf_to_96p, get_next_span_days
from api.model.data import submit_power_to_server
from api.io.logger import setup_logger
from api.io.messenger import messenger_feishu
from df_to_mongo import dataframe_to_mongodb,smooth_daytime_window

logger = setup_logger(__name__)
GLOBAL_TIMESTAMP = None
CUR_DATE =  datetime.datetime.today().date().strftime('%Y-%m-%d %H%M%S')


def set_global_timestamp(timestamp):
    global GLOBAL_TIMESTAMP
    GLOBAL_TIMESTAMP = timestamp


def daily_energy_job(
    model_type: str,
    train: bool,
    label_list: Sequence[str],
    feature_list: Sequence[str],
    trade_type: str,
    local_dir: str,
    load_strictly: bool,
    get_label: bool,
    submit_result: bool,
    meteo_scope: str = 'station',
    start_date: str = None,
    end_date: str = None,
    start_span: int = None,
    end_span: int = None,
    **kwargs,
):
    try:
        assert (start_date is not None) or (start_span is not None)
        assert (end_date is not None) or (end_span is not None)

        pipeline_version = GLOBAL_TIMESTAMP
        if start_date is None:
            cur_date = datetime.datetime.strptime(pipeline_version, "%Y%m%d-%H%M%S")
            start_date = (cur_date.date() + datetime.timedelta(days=start_span)).strftime('%Y-%m-%d')
        if end_date is None:
            cur_date = datetime.datetime.strptime(pipeline_version, "%Y%m%d-%H%M%S")
            end_date = (cur_date.date() + datetime.timedelta(days=end_span)).strftime('%Y-%m-%d')
        output_start_date = start_date
        model_start_date = start_date


        if not train:
            model_start_date = (
                datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
                + datetime.timedelta(days=-4)
            ).strftime('%Y-%m-%d')

        logger.info(
            "[job] model: {}, train: {}, labels: {}, date: {}~{}, pipeline: {}, meteo_source: {}".format(
                model_type, train, label_list, start_date, end_date, pipeline_version, kwargs.get('meteo_source', 'ENS_EXTEND')
            )
        )
        if not train:
            logger.info(
                "[job] 预测拼接历史窗口: 模型取数 {}~{}，结果输出 {}~{}".format(
                    model_start_date, end_date, output_start_date, end_date
                )
            )
        # 初始化模型流程
        pipeline = ModelPipeline(
            model_type=model_type,
            train=train,
            label_list=label_list,
            feature_list=feature_list,
            trade_type=trade_type,
            start_date=model_start_date,
            end_date=end_date,
            pipeline_version=pipeline_version,
            local_dir=local_dir,
            load_strictly=load_strictly,
            get_label=get_label,
            meteo_scope=meteo_scope,
            **kwargs,
        )

        pdf_file = f'./output/pdf_rst_jiangsu.csv'

        # 执行模型流程
        pipeline.run()

        # 处理预测结果, 计算准确率
        predict_results = pipeline.predict_results

        pdf_rst = None
        for label in label_list:
            pdf_predict: pd.DataFrame = predict_results[label]
            # 预测模式下，只保留原始 start_date 及之后的结果；
            # start_date 前 4 天只是给时序模型提供冷启动历史窗口，不参与输出和评估。
            if not train:
                pdf_predict = pdf_predict.copy()
                pdf_predict['date_time'] = pd.to_datetime(pdf_predict['date_time'])
                output_start_ts = pd.to_datetime(output_start_date)
                before_count = len(pdf_predict)
                pdf_predict = pdf_predict[pdf_predict['date_time'] >= output_start_ts].copy()
                logger.info(
                    "{} 预测结果已截断至 {}，过滤历史窗口 {} 行，剩余 {} 行".format(
                        label, output_start_date, before_count - len(pdf_predict), len(pdf_predict)
                    )
                )
            # print(pdf_predict)
            if get_label:
                # 计算MAPE的特殊处理的阈值
                threshold = 1000
                acc = calculate_accuracy(
                    val_pred=pdf_predict["{}_pred".format(label)],
                    val_true=pdf_predict[label],
                    threshold=threshold,
                )
                logger.info("{} 预测准确率: {}".format(label, acc))
            if pdf_rst is None:
                pdf_rst = pdf_predict[
                    ["date_time", "version", "{}_pred".format(label),label] #,label
                ].copy()
            else:
                pdf_rst = pdf_rst.merge(
                    pdf_predict[
                        [
                            "date_time",
                            "{}_pred".format(label),
                            label,
                        ]## "version","version_diff",
                    ],
                    on=["date_time"],##"version", "version_diff",
                    how="left",
                )
            pdf_eval = pdf_rst[['date_time', 'new_energy_wind_ahead', 'new_energy_wind_ahead_pred']].copy()
            pdf_eval['error'] = pdf_eval['new_energy_wind_ahead_pred'] - pdf_eval['new_energy_wind_ahead']
            pdf_eval['abs_error'] = pdf_eval['error'].abs()
            pdf_eval['hour'] = pd.to_datetime(pdf_eval['date_time']).dt.hour

            print("Bias:", pdf_eval['error'].mean())
            print("MAE:", pdf_eval['abs_error'].mean())
            print("按功率分箱的 MAE:")
            print(pd.qcut(pdf_eval['new_energy_wind_ahead'], 5).value_counts())
            print(pdf_eval.groupby(pd.qcut(pdf_eval['new_energy_wind_ahead'], 5))['abs_error'].mean())

            pdf_rst_sorted = pdf_rst.copy()
            pdf_rst_sorted['date_time'] = pd.to_datetime(pdf_rst['date_time'])
            pdf_rst_sorted = pdf_rst_sorted.sort_values(by='date_time', ascending=True)

        pdf_rst_sorted = pdf_rst_sorted.rename(columns={'new_energy_wind_ahead_pred': 'new_energy_wind_pred', 'new_energy_solar_ahead_pred': 'new_energy_solar_pred'})
        # pdf_rst_sorted.to_csv(pdf_file,index = False)

        pdf_rst_sorted['new_energy_wind_pred'] = smooth_gaussian(pdf_rst_sorted['new_energy_wind_pred'])
        # pdf_rst_sorted['new_energy_solar_pred'] = smooth_gaussian(pdf_rst_sorted['new_energy_solar_pred'])
        pdf_rst_sorted.to_csv(pdf_file,index = False)

        # pdf_rst_sorted['new_energy_all'] = pdf_rst_sorted['new_energy_wind_pred']+pdf_rst_sorted['new_energy_solar_pred']
        # pdf_rst_sorted['new_energy_all'] = np.round(pdf_rst_sorted['new_energy_all'], 3)
        # pdf_rst_sorted['new_energy_wind_pred'] = np.round(pdf_rst_sorted['new_energy_wind_pred'], 3)
        # pdf_rst_sorted['new_energy_solar_pred'] = np.round(pdf_rst_sorted['new_energy_solar_pred'], 3)

        # pdf_rst_sorted['date'] = pdf_rst_sorted['date_time'].dt.strftime('%Y-%m-%d')
        # pdf_rst_sorted['time'] = pdf_rst_sorted['date_time'].dt.strftime('%H:%M:%S')
        
        # CUR_DATE = datetime.datetime.now() + datetime.timedelta(hours=8)
        # if 12 <= CUR_DATE.hour:
        #     pdf_rst_sorted['version'] = pd.to_datetime(CUR_DATE.date().strftime('%Y-%m-%d') + ' 18:00:00')
        # else:  
        #     pdf_rst_sorted['version'] = pd.to_datetime(CUR_DATE.date().strftime('%Y-%m-%d') + ' 08:00:00')

        # pdf_rst_sorted['version_date'] = pdf_rst_sorted['version'].dt.strftime('%Y-%m-%d')
        # pdf_rst_sorted['version_time'] = pdf_rst_sorted['version'].dt.strftime('%H:%M:%S')
        
        # pdf_rst_sorted['create_time'] = CUR_DATE
        # pdf_rst_sorted['update_time'] = CUR_DATE

        # pdf_rst_sorted = pdf_rst_sorted.iloc[:1056,:]##10天预测

        # if train==False:

        #     data_m = pdf_rst_sorted.copy()
        #     data_m = data_m[['date_time','create_time','date','new_energy_all','new_energy_solar_pred','new_energy_wind_pred','time','update_time','version','version_date','version_time']]

        #     data_m['date_time'] = pd.to_datetime(data_m['date_time'])
        #     data_m['new_energy_solar_pred'] = data_m['new_energy_solar_pred'] * 1.12 #2025年装机(4/3)， ##2026年光伏江北/江南1.12
        #     # dataframe_to_mongodb(data_m)

        #     pdf_rst_sorted.to_csv(pdf_file,index = False)

        # other_text = ""
        # if train:
        #     other_text = "训练"
        # else:
        #     other_text = "预测"
        # other_text += f"-{kwargs.get('meteo_source', 'ENS_EXTEND')}-{meteo_scope}"

        # messenger_feishu.collect(
        #     run_version_time = None,
        #     target_name = "new_energy",
        #     task_name=kwargs.get('task_name', 'by_day'),
        #     span_day=[start_date, end_date],
        #     region = "江苏",
        #     failure = '成功',
        #     other_text=other_text + f'预测文件输出{pdf_file}'
        # )
        # messenger_feishu.send()
    except Exception as e:
        other_text = ""
        if train:
            other_text = "训练"
        else:
            other_text = "预测"
        other_text += f"-{kwargs.get('meteo_source', 'ENS_EXTEND')}-{meteo_scope}"

        messenger_feishu.collect(
            run_version_time = None,
            target_name = "new_energy",
            task_name=kwargs.get('task_name', 'by_day'),
            span_day=[start_date, end_date],
            region = "江苏",
            failure = '失败',
            other_text=other_text
        )
        messenger_feishu.send()
        logger.exception(e)
    



def longterm_energy_job(
    model_type: str,
    train: bool,
    label_list: Sequence[str],
    feature_list: Sequence[str],
    trade_type: str,
    local_dir: str,
    load_strictly: bool,
    get_label: bool,
    submit_result: bool,
    meteo_scope: str = 'station',
    start_date: str = None,
    end_date: str = None,
    start_span: int = None,
    end_span: int = None,
    **kwargs,
):
    try:
        pipeline_version = GLOBAL_TIMESTAMP
        task_name = kwargs.get('task_name', 'by_day')
        if start_date is None and start_span is not None:
            cur_date = datetime.datetime.strptime(pipeline_version, "%Y%m%d-%H%M%S")
            start_date = (cur_date.date() + datetime.timedelta(days=start_span)).strftime('%Y-%m-%d')
        if end_date is None and end_span is not None:
            cur_date = datetime.datetime.strptime(pipeline_version, "%Y%m%d-%H%M%S")
            end_date = (cur_date.date() + datetime.timedelta(days=end_span)).strftime('%Y-%m-%d')
        if not start_date and not end_date and not start_span and not end_span:
            start_date, end_date = get_next_span_days(task_name)
        if kwargs.get("version_start_date", None) is not None and kwargs.get("version_end_date", None) is None:
            cur_date = datetime.datetime.strptime(pipeline_version, "%Y%m%d-%H%M%S")
            version_end_date = (cur_date.date() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
            kwargs.update({"version_end_date": version_end_date})
        
        logger.info(
            "[job_{}] model: {}, train: {}, labels: {}, date: {}~{}, pipeline: {}, meteo_source: {}".format(
                task_name, model_type, train, label_list, start_date, end_date, pipeline_version, kwargs.get('meteo_source', 'ENS_EXTEND')
            )
        )


        # 初始化模型流程
        pipeline = ModelPipeline(
            model_type=model_type,
            train=train,
            label_list=label_list,
            feature_list=feature_list,
            trade_type=trade_type,
            start_date=start_date,
            end_date=end_date,
            pipeline_version=pipeline_version,
            local_dir=local_dir,
            load_strictly=load_strictly,
            get_label=get_label,
            meteo_scope=meteo_scope,
            **kwargs,
        )


        # 执行模型流程
        pipeline.run()

        # 处理预测结果, 计算准确率
        predict_results = pipeline.predict_results
        pdf_rst = None
        for label in label_list:
            pdf_predict: pd.DataFrame = predict_results[label]

            if get_label:
                # 计算MAPE的特殊处理的阈值
                threshold = 1000

                acc = calculate_accuracy(
                    val_pred=pdf_predict["{}_pred".format(label)],
                    val_true=pdf_predict[label],
                    threshold=threshold,
                )

                logger.info("{} 预测准确率: {}".format(label, acc))

            if pdf_rst is None:
                pdf_rst = pdf_predict[
                    ["date_time", "version", "version_diff", "meteo_source", "{}_pred".format(label)]
                ].copy()
            else:
                pdf_rst = pdf_rst.merge(
                    pdf_predict[
                        [
                            "date_time",
                            "version",
                            "version_diff",
                            "meteo_source",
                            "{}_pred".format(label),
                        ]
                    ],
                    on=["date_time", "version", "version_diff", "meteo_source"],
                    how="left",
                )

        # if submit_result:
        #     if task_name == 'by_day':
        #         pdf_rst = convert_pdf_to_96p(
        #             pdf_data=pdf_rst,
        #             interp_cols=[f"{item}_pred" for item in label_list],
        #             ffill_list=["version_diff", "version", "meteo_source"],
        #             version_count=kwargs.get("version_num", 0),
        #         )
        #         logger.info("Finish interpolating values after prediction.")
        #     logger.debug(pdf_rst.head())

        #     use_ahead = False
        #     if feature_list:
        #         use_ahead = True
        #     pdf_rst['model_log_name'] = pipeline_version
        #     submit_power_to_server(pdf_rst, trade_type, label_list, use_ahead, task_name=task_name)

        messenger_feishu.collect(
            run_version_time = None,
            target_name = "new_energy",
            task_name=kwargs.get('task_name', 'by_day'),
            span_day=[start_date, end_date],
            region = "jiangsu",
            failure = '成功',
            other_text=other_text + f'预测文件输出{pdf_file}'
        )
        messenger_feishu.send()
    except Exception as e:
        messenger_feishu.collect(
            run_version_time = None,
            target_name = "new_energy",
            task_name=kwargs.get('task_name', 'by_day'),
            span_day=[start_date, end_date],
            region = "jiangsu",
            failure = '失败',
            other_text=""
        )
        messenger_feishu.send()
        logger.exception(e)