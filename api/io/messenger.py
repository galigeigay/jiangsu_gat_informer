import base64
from datetime import datetime
import hashlib
import hmac
import json
import requests
import time
from typing import List

FEISHU_URL = 'https://open.feishu.cn/open-apis/bot/v2/hook/08acc684-8348-4755-8cff-995ac04c9626'
FEISHU_SECRET = ''

PHONE = '12345678901'


def gen_sign_feishu(timestamp, secret):
    # 拼接timestamp和secret
    string_to_sign = '{}\n{}'.format(timestamp, secret)
    hmac_code = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()

    # 对结果进行base64处理
    sign = base64.b64encode(hmac_code).decode('utf-8')
    return sign

task_mapping = {
    'new_energy': '新能源',
    'new_energy_solar': '实时光',
    'new_energy_wind': '实时风',
    'ahead_new_energy_solar': '日前光',
    'ahead_new_energy_wind': '日前风',
    'default': '其他'
}

span_day_mapping = {
    'pred_d_1': '1日',
    'pred_d_5': '5日',
    'pred_d_10': '旬',
    'pred_d_30': '月',
    'pred_d_30_10': '月分旬',
    'default': '其他'
}

class Messenger(object):
    def __init__(self,
                 webhook_url: str,
                 webhook_secret: str,
                 fail_at_phone: str,
                 always_at_phone: bool = False,
                 ):


        self.run_time_type = str(datetime.now().hour)

        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.fail_at_phone = fail_at_phone
        self.need_at_phone = always_at_phone

        self.msg_list = []


    def collect(self,
                run_version_time: str,
                target_name: str,
                task_name: str,
                span_day: List[str],
                region: str,
                failure: str,
                other_text: str=''
                ):
        
        if not run_version_time:
            run_version_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg_str = '{}-{} {} {} {} {}~{} {} {} {}'.format(
            run_version_time,
            self.run_time_type,
            region,
            task_mapping[target_name],
            task_name,
            span_day[0],
            span_day[1],
            other_text,
            failure,
            cur_time,
        )
        self.msg_list.append(msg_str)

        return

    def send(self, send_target: str='Feishu'):
        content = "\n".join(self.msg_list)

        if send_target == 'Feishu':
            feishu_timestamp = str(int(time.time()))
            if self.need_at_phone:
                at_content = '<at user_id = \"all\">所有人</at>'
            else:
                at_content = ''
            dict = {
                    "timestamp": feishu_timestamp,       # 时间戳
                    "sign": gen_sign_feishu(feishu_timestamp, self.webhook_secret),  # 得到的签名字符串
                    "msg_type": "text",
                    "content": {
                        "text": '{} {}'.format(at_content, content)
                    }
                }
            req = requests.post(url = self.webhook_url, headers={'Content-Type': 'application/json'}, data=json.dumps(dict))
        else:

            raise ValueError('Cannot recognize the send_target type')

        return


messenger_feishu = Messenger(
    webhook_url=FEISHU_URL,
    webhook_secret=FEISHU_SECRET,
    fail_at_phone=PHONE,
    always_at_phone=False,
)