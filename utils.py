"""
工具函数模块
"""

import time
import random
import hashlib
import urllib.parse
from functools import reduce
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bilibili_spider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def random_sleep(min_sec: float = 1, max_sec: float = 3):
    sleep_time = random.uniform(min_sec, max_sec)
    time.sleep(sleep_time)


def get_md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_mixin_key(orig: str) -> str:
    mixin_key_enc_tab = [
        46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
        27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
        37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
        22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52
    ]
    return reduce(lambda s, i: s + orig[i], mixin_key_enc_tab, '')[:32]


def encWbi(params: Dict[str, Any], img_key: str, sub_key: str) -> Dict[str, Any]:
    mixin_key = get_mixin_key(img_key + sub_key)
    curr_time = round(time.time())
    params['wts'] = curr_time
    params = dict(sorted(params.items()))
    params = {
        k: ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
        for k, v in params.items()
    }
    query = urllib.parse.urlencode(params)
    wbi_sign = get_md5(query + mixin_key)
    params['w_rid'] = wbi_sign
    return params


def format_duration(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_number(num: int) -> str:
    if num >= 100000000:
        return f"{num / 100000000:.1f}亿"
    elif num >= 10000:
        return f"{num / 10000:.1f}万"
    return str(num)


def safe_get(data: Dict, *keys, default=None):
    for key in keys:
        try:
            data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


def calculate_engagement_rate(stat: Dict) -> float:
    view = stat.get('view', 0)
    if view == 0:
        return 0.0
    engagement = (stat.get('like', 0) + stat.get('coin', 0) +
                  stat.get('favorite', 0) + stat.get('share', 0))
    return round(engagement / view, 4)


def calculate_completion_rate_proxy(stat: Dict, duration: int) -> float:
    view = stat.get('view', 0)
    if view == 0:
        return 0.0
    completion_proxy = stat.get('coin', 0) + stat.get('favorite', 0)
    return round(completion_proxy / view, 4)
