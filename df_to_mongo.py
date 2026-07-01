import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
try:
    from pymongo.errors import AuthenticationError
except ImportError:
    # 旧版本pymongo中AuthenticationError可能属于OperationFailure
    from pymongo.errors import OperationFailure as AuthenticationError
import logging
import datetime
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_mongodb(host, port, db_name, username, password,auth_source):

    try:
        # 创建MongoDB客户端，支持认证
        client_kwargs = {
            'host': host,
            'port': port
        }
        
        # 如果提供了用户名和密码，则添加认证信息
        if username and password:
            client_kwargs['username'] = username
            client_kwargs['password'] = password
            client_kwargs['authSource'] = auth_source
        
        # 创建客户端
        client = MongoClient(**client_kwargs)
        
        # 验证连接
        client.admin.command('ping')
        logger.info("成功连接到MongoDB")
        
        # 返回数据库对象
        return client[db_name]
    
    except AuthenticationError:
        logger.error("MongoDB认证失败，请检查用户名和密码")
        return None
    except ConnectionFailure:
        logger.error("无法连接到MongoDB服务器，请检查主机和端口")
        return None
    except Exception as e:
        logger.error(f"连接MongoDB时发生错误: {str(e)}")
        return None


def dataframe_to_mongodb(df, collection_name='ods_jiangsu_new_energy_load_forecast'):

    if df.empty:
        logger.warning("DataFrame为空，不执行写入操作")
        return False
    
    # MongoDB连接信息，包含认证信息
    # predict
    # SfAeScRuJse7ME!
    mongo_info = {
        'host': '1.94.208.154',
        'port': 8635,
        'user': 'predict',
        'password': 'SfAeScRuJse7ME!',
        'db_name': 'yn_ams_dw',  # 指定要使用的数据库名
        'auth_source': 'admin'    
    }
    
    # 连接到MongoDB，包含认证信息
    db = connect_to_mongodb(
        host=mongo_info['host'],
        port=mongo_info['port'],
        db_name=mongo_info['db_name'],
        username=mongo_info['user'],
        password=mongo_info['password'],
        auth_source=mongo_info['auth_source']
    )
    
    if db is None:
        return False
    
    try:
        # 获取集合（如果不存在会自动创建）
        collection = db[collection_name]

        # 将DataFrame转换为字典列表
        data = df.to_dict('records')
        
        if not data:
            logger.warning("没有数据可写入MongoDB")
            return True
        
        # 记录成功和失败的数量
        success_count = 0
        fail_count = 0
        
        # 逐条处理数据，使用date_time作为查询条件
        for item in data:
            if 'date_time' not in item:
                logger.warning("数据中缺少date_time字段，跳过此条记录")
                fail_count += 1
                continue
                
            # 以date_time为条件，存在则更新，不存在则插入
            # filter_query = {'date_time': item['date_time']}
            filter_query = {'date_time': item['date_time'],'version': item['version']}
            update_query = {'$set': item}
            
            try:
                result = collection.update_one(filter_query, update_query, upsert=True)
                if result.upserted_id:
                    logger.debug(f"新增记录: {item['date_time']}")
                else:
                    logger.debug(f"更新记录: {item['date_time']}")
                success_count += 1
            except PyMongoError as e:
                logger.error(f"处理记录 {item['date_time']} 时出错: {str(e)}")
                fail_count += 1
        
        logger.info(f"处理完成 - 成功: {success_count} 条, 失败: {fail_count} 条")
        return success_count > 0 or fail_count == 0
    
    except PyMongoError as e:
        logger.error(f"写入MongoDB时发生错误: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"处理数据时发生错误: {str(e)}")
        return False

def main(pdf_upload_info):
    """主函数，示例如何使用上述功能"""
    try:
        # 读取CSV文件
        csv_path = pdf_upload_info
        logger.info(f"开始读取CSV文件: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"成功读取CSV文件，共 {len(df)} 条记录")
        
        # 处理日期列（根据实际CSV中的日期列名调整）
        if 'date_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date_time'])
            logger.info("已将date_time列转换为MongoDB日期格式")
        else:
            logger.warning("CSV文件中未找到date_time列，跳过日期转换")
        
        # 写入MongoDB
        success = dataframe_to_mongodb(df)
        
        if success:
            logger.info("数据处理完成")
        else:
            logger.error("数据处理失败")
            
    except FileNotFoundError:
        logger.error(f"CSV文件未找到: {csv_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"CSV文件为空: {csv_path}")
    except pd.errors.ParserError:
        logger.error(f"CSV文件解析错误: {csv_path}")
    except Exception as e:
        logger.error(f"主程序发生错误: {str(e)}")

def resample_to_15min(df, time_column='timestamp'):
    try:
        if time_column not in df.columns:
            logger.error(f"数据中不存在时间列: {time_column}")
            return None
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
        
        df_copy['date'] = df_copy[time_column].dt.date
        result_list = []
        all_dates = sorted(df_copy['date'].unique())
        
        for date in all_dates:
            next_date = date + datetime.timedelta(days=1)
            
            # 取当天 + 次日数据作为边界锚点
            boundary_group = df_copy[
                (df_copy['date'] == date) |
                (df_copy['date'] == next_date)
            ].copy()
            
            # version 取当天的值
            version_value = None
            day_group = df_copy[df_copy['date'] == date]
            if 'version' in day_group.columns and not day_group['version'].isna().all():
                version_value = day_group['version'].dropna().iloc[0]
            else:
                logger.warning(f"日期{date}无有效version值，version列填充为NaN")
            
            # 索引延伸到次日 00:00（共97个点）
            start_time = datetime.datetime.combine(date, datetime.time.min)
            end_time = start_time + datetime.timedelta(days=1)
            full_15min_index = pd.date_range(start=start_time, end=end_time, freq='15min')
            
            boundary_group = boundary_group.set_index(time_column)
            boundary_group = boundary_group.drop(columns=['date'], errors='ignore')
            
            # day_resampled = boundary_group.resample('15min').interpolate(method='linear')

            # 只对数值列做 resample，排除 object 和 datetime 列
            numeric_cols_raw = boundary_group.select_dtypes(include=['number']).columns.tolist()

            day_resampled = boundary_group[numeric_cols_raw].resample('15min').mean()
            day_resampled = day_resampled.infer_objects(copy=False).interpolate(method='linear')

            # print(boundary_group.select_dtypes(include=['object']).columns.tolist())
            # print(boundary_group.select_dtypes(include=['datetime64']).columns.tolist())
            # print(boundary_group.info())
            # exit()
            day_full = day_resampled.reindex(full_15min_index)
            
            numeric_cols = day_full.select_dtypes(include=['number']).columns
            day_full[numeric_cols] = (
                day_full[numeric_cols]
                .interpolate(method='linear', limit_direction='both')
                .bfill() 
            )
            
            # 裁剪掉次日 00:00 边界点，保留当天 96 个点（00:00 ~ 23:45）
            day_full = day_full.iloc[:-1]
            
            if 'version' not in day_full.columns:
                day_full = day_full.copy()
                day_full['version'] = None
            day_full['version'] = version_value
            
            day_full = day_full.reset_index(names=time_column)
            
            if len(day_full) != 96:
                logger.warning(f"日期{date}行数异常，实际{len(day_full)}行（预期96行）")
            
            result_list.append(day_full)
        
        if not result_list:
            logger.error("无有效日期数据可处理")
            return None
        
        df_final = pd.concat(result_list, ignore_index=True)
        logger.info(f"所有日期处理完成: 原始{len(df)}条记录 -> 新{len(df_final)}条记录")
        
        return df_final
    
    except Exception as e:
        logger.error(f"重采样过程中发生错误: {str(e)}")
        return None


def resample_to_15min_v1(df, time_column='timestamp'):
    try:
        if time_column not in df.columns:
            logger.error(f"数据中不存在时间列: {time_column}")
            return None
        
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[time_column]):
            df_copy[time_column] = pd.to_datetime(df_copy[time_column])
            # logger.info(f"已将'{time_column}'列转换为datetime类型")
        
        df_copy['date'] = df_copy[time_column].dt.date  # 新增日期列用于分组
        result_list = []
        
        for date, day_group in df_copy.groupby('date'):
            # logger.info(f"开始处理日期: {date}")
            
            start_time = datetime.datetime.combine(date, datetime.datetime.min.time())
            end_time = start_time + datetime.timedelta(days=1) - datetime.timedelta(minutes=15)
            full_15min_index = pd.date_range(start=start_time, end=end_time, freq='15min')
            
            version_value = None
            if 'version' in day_group.columns and not day_group['version'].isna().all():
                version_value = day_group['version'].dropna().iloc[0]
                # logger.info(f"日期{date}的统一version值: {version_value}")
            else:
                logger.warning(f"日期{date}无有效version值，version列填充为NaN")
            
            day_group = day_group.set_index(time_column)
            day_resampled = day_group.resample('15min').interpolate(method='linear')
            day_full = day_resampled.reindex(full_15min_index)
            
            numeric_cols = day_full.select_dtypes(include=['number']).columns
            day_full[numeric_cols] = day_full[numeric_cols].interpolate(method='linear').fillna(method='bfill')
            
            if 'version' not in day_full.columns:
                day_full['version'] = None
            day_full['version'] = version_value
            
            day_full = day_full.reset_index(names=time_column)
            day_full = day_full.drop(columns=['date'], errors='ignore')  # 删除临时日期列
            
            if len(day_full) != 96:
                logger.warning(f"日期{date}行数异常，实际{len(day_full)}行（预期96行）")
            # else:
            #     logger.info(f"日期{date}处理完成，生成96个15分钟时次")
            
            result_list.append(day_full)
        
        if not result_list:
            logger.error("无有效日期数据可处理")
            return None
        
        df_final = pd.concat(result_list, ignore_index=True)
        logger.info(f"所有日期处理完成: 原始{len(df)}条记录 -> 新{len(df_final)}条记录")
        
        return df_final
    
    except Exception as e:
        logger.error(f"重采样过程中发生错误: {str(e)}")
        return None

def resample_to_15min_v0(df, time_column='timestamp'):

    try:
        if time_column not in df.columns:
            logger.error(f"数据中不存在时间列: {time_column}")
            return None
        
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
            logger.info(f"已将'{time_column}'列转换为datetime类型")
        
        df_resample = df.set_index(time_column).copy()
        
        time_diff = df_resample.index.to_series().diff().mode().iloc[0]
        if time_diff != pd.Timedelta(hours=1):
            logger.warning(f"原始数据的时间间隔不是1小时，实际为: {time_diff}")
        
        df_15min = df_resample.resample('15min').asfreq()
        df_15min = df_15min.interpolate(method='linear')
        
        df_15min = df_15min.reset_index()
        
        logger.info(f"数据重采样完成: 原始{len(df)}条记录 -> 新{len(df_15min)}条记录")
        return df_15min
    
    except Exception as e:
        logger.error(f"重采样过程中发生错误: {str(e)}")
        return None

from scipy.signal import savgol_filter
import numpy as np 

def smooth_daytime_window(
    df: pd.DataFrame,
    columns: list[str] = ["new_energy_all", "new_energy_solar_pred"],
    start_hour: int = 9,
    end_hour: int = 15,
    window_length: int = 11,
    polyorder: int = 2,
    datetime_col: str = "date_time",
) -> pd.DataFrame:

    result = df.copy()
    result[datetime_col] = pd.to_datetime(result[datetime_col])
    result["_date"] = result[datetime_col].dt.date
 
    for col in columns:
        if col not in result.columns:
            raise KeyError(f"列不存在: {col}")
 
        for date, group in result.groupby("_date"):
            hour = group[datetime_col].dt.hour + group[datetime_col].dt.minute / 60
            mask = (hour >= start_hour) & (hour < end_hour)
            idx = group.index[mask]
 
            if len(idx) < window_length:
                continue
 
            seg = result.loc[idx, col].to_numpy(dtype=float)
            smoothed = savgol_filter(seg, window_length=window_length, polyorder=polyorder)
            result.loc[idx, col] = np.clip(smoothed, 0, None)
 
    result.drop(columns=["_date"], inplace=True)
    return result


if __name__ == "__main__":

    date_now = datetime.datetime.now().strftime('%Y-%m-%d')
    m_date = datetime.datetime.now().strftime('%Y%m%d')

    st = (datetime.datetime.now()+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    ed = (datetime.datetime.now()+datetime.timedelta(days=11)).strftime('%Y-%m-%d')

    model_time= date_now

    file_output_path = '/data00/chenjiajun/20250711/daily_test/jiangsu_power_cronjob_v1/output'

    pdf_upload_info = pd.read_csv(os.path.join(file_output_path, m_date[:6], f'pdf_rst_{m_date}_{st}_{ed}.csv'))


    main(pdf_upload_info)
    