"""
B站API调用封装
"""

import requests
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Set
from utils import logger, random_sleep, encWbi, safe_get
from config import (
    HEADERS, POPULAR_API, RANKING_API, USER_INFO_API, NAV_API,
    MAX_RETRIES, TIMEOUT, REQUEST_DELAY_MIN, REQUEST_DELAY_MAX
)


class BilibiliAPI:
    """B站API调用类"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.img_key = None
        self.sub_key = None
        self.failed_mids: Set[int] = set()
        self.success_count = 0
        self.fail_count = 0
        self._init_wbi_keys()

    def _init_wbi_keys(self):
        try:
            response = self.session.get(NAV_API, timeout=TIMEOUT)
            data = response.json()
            if data['code'] == 0:
                wbi_img = safe_get(data, 'data', 'wbi_img', 'img_url', default='')
                wbi_sub = safe_get(data, 'data', 'wbi_img', 'sub_url', default='')
                self.img_key = wbi_img.rsplit('/', 1)[-1].split('.')[0] if wbi_img else None
                self.sub_key = wbi_sub.rsplit('/', 1)[-1].split('.')[0] if wbi_sub else None
                if self.img_key and self.sub_key:
                    logger.info(f"Wbi密钥初始化成功")
                else:
                    logger.warning("Wbi密钥获取失败")
            else:
                logger.warning(f"获取Wbi密钥失败: {data.get('message')}")
        except Exception as e:
            logger.error(f"初始化Wbi密钥异常: {e}")

    def _request_with_retry(self, url: str, params: Dict = None,
                            method: str = 'GET', is_user_api: bool = False) -> Optional[Dict]:
        for attempt in range(MAX_RETRIES):
            try:
                if is_user_api:
                    random_sleep(3, 6)
                else:
                    random_sleep(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)

                headers = self.session.headers.copy()
                if is_user_api:
                    headers['X-Requested-With'] = 'XMLHttpRequest'

                if method == 'GET':
                    response = self.session.get(url, params=params, headers=headers, timeout=TIMEOUT)
                else:
                    response = self.session.post(url, json=params, headers=headers, timeout=TIMEOUT)

                response.raise_for_status()
                data = response.json()

                if data.get('code') == 0:
                    return data
                elif data.get('code') == -352:
                    logger.warning(f"⚠️ 触发风控 (尝试 {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(random.uniform(15, 30))
                else:
                    logger.warning(f"API返回错误: code={data.get('code')}, message={data.get('message')}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{MAX_RETRIES})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败 (尝试 {attempt + 1}/{MAX_RETRIES})")

            if attempt < MAX_RETRIES - 1:
                random_sleep(5, 10)

        logger.error(f"请求最终失败: {url}")
        return None

    def get_popular_videos(self, page: int = 1, page_size: int = 20) -> List[Dict]:
        logger.info(f"正在获取综合热门第{page}页...")
        params = {'ps': page_size, 'pn': page}
        data = self._request_with_retry(POPULAR_API, params)
        if data and 'data' in data and 'list' in data['data']:
            videos = data['data']['list']
            logger.info(f"成功获取{len(videos)}条综合热门视频")
            return videos
        return []

    def get_ranking_videos(self, rid: int, page_size: int = 50) -> List[Dict]:
        logger.info(f"正在获取分区{rid}的热门视频...")
        params = {'rid': rid, 'type': 'all'}
        data = self._request_with_retry(RANKING_API, params)
        if data and 'data' in data and 'list' in data['data']:
            videos = data['data']['list'][:page_size]
            logger.info(f"成功获取分区{rid}的{len(videos)}条热门视频")
            return videos
        return []

    def get_user_info(self, mid: int) -> Optional[Dict]:
        if not self.img_key or not self.sub_key:
            self.failed_mids.add(mid)
            return None
        params = {'mid': mid}
        signed_params = encWbi(params, self.img_key, self.sub_key)
        data = self._request_with_retry(USER_INFO_API, signed_params, is_user_api=True)
        if data and 'data' in data:
            user_data = data['data']
            return {
                'mid': mid,
                'name': user_data.get('name', ''),
                'level': safe_get(user_data, 'level', default=0),
                'fans': safe_get(user_data, 'follower', default=0),
                'vip_type': safe_get(user_data, 'vip', 'type', default=0),
                'vip_status': safe_get(user_data, 'vip', 'status', default=0),
                'official_verify': safe_get(user_data, 'official', 'type', default=-1),
            }
        self.failed_mids.add(mid)
        return None

    def batch_get_user_info(self, mids: List[int], batch_size: int = 50) -> Dict[int, Dict]:
        logger.info(f"开始批量获取{len(mids)}个UP主信息...")
        user_info_dict = {}
        self.success_count = 0
        self.fail_count = 0
        consecutive_fails = 0

        for i, mid in enumerate(mids, 1):
            logger.info(f"获取UP主信息进度: {i}/{len(mids)} (成功:{self.success_count}, 失败:{self.fail_count})")

            if i % batch_size == 0 and i < len(mids):
                extra_sleep = random.uniform(10, 20)
                logger.info(f"已完成{i}个请求，休息{extra_sleep:.1f}秒...")
                time.sleep(extra_sleep)

            user_info = self.get_user_info(mid)
            if user_info:
                user_info_dict[mid] = user_info
                self.success_count += 1
                consecutive_fails = 0
            else:
                self.fail_count += 1
                consecutive_fails += 1

                if consecutive_fails >= 5:
                    logger.error(f"⚠️ 连续失败{consecutive_fails}次！")
                    print(f"\n连续{consecutive_fails}次失败，是否继续？")
                    print("1. 继续  2. 停止")
                    choice = input("选择 (1/2): ").strip()
                    if choice == "2":
                        break
                    time.sleep(random.uniform(30, 60))
                    consecutive_fails = 0

        logger.info(f"UP主信息获取完成: 成功{self.success_count}/{len(mids)}")

        if self.failed_mids:
            self._save_failed_mids()

        return user_info_dict

    def _save_failed_mids(self, filename: str = 'failed_mids.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'failed_mids': list(self.failed_mids),
                    'count': len(self.failed_mids),
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 失败的mid列表已保存到: {filename} (共{len(self.failed_mids)}个)")
        except Exception as e:
            logger.error(f"保存失败mid列表时出错: {e}")

    def load_failed_mids(self, filename: str = 'failed_mids.json') -> List[int]:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('failed_mids', [])
        except FileNotFoundError:
            return []

    def get_statistics(self) -> Dict:
        return {
            'success_count': self.success_count,
            'fail_count': self.fail_count,
            'failed_mids': list(self.failed_mids),
            'failed_count': len(self.failed_mids)
        }
