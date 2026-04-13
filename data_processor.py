"""
数据处理与转换模块
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict
from utils import logger, format_duration, calculate_engagement_rate, calculate_completion_rate_proxy


class DataProcessor:
    """数据处理类"""

    @staticmethod
    def extract_video_info(video_raw: Dict, source: str = 'popular') -> Dict:
        stat = video_raw.get('stat', {})
        owner = video_raw.get('owner', {})
        duration = video_raw.get('duration', 0)

        video_info = {
            'bvid': video_raw.get('bvid', ''),
            'aid': video_raw.get('aid', 0),
            'title': video_raw.get('title', ''),
            'desc': video_raw.get('desc', ''),
            'duration': duration,
            'duration_formatted': format_duration(duration),
            'pubdate': video_raw.get('pubdate', 0),
            'pubdate_formatted': datetime.fromtimestamp(
                video_raw.get('pubdate', 0)
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'tname': video_raw.get('tname', ''),
            'tid': video_raw.get('tid', 0),
            'view': stat.get('view', 0),
            'danmaku': stat.get('danmaku', 0),
            'reply': stat.get('reply', 0),
            'favorite': stat.get('favorite', 0),
            'coin': stat.get('coin', 0),
            'share': stat.get('share', 0),
            'like': stat.get('like', 0),
            'mid': owner.get('mid', 0),
            'owner_name': owner.get('name', ''),
            'owner_level': None,
            'owner_fans': None,
            'owner_vip_type': None,
            'owner_official_verify': None,
            'data_source': source,
            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        video_info['engagement_rate'] = calculate_engagement_rate(stat)
        video_info['completion_rate_proxy'] = calculate_completion_rate_proxy(stat, duration)

        view = stat.get('view', 0)
        if view > 0:
            video_info['like_rate'] = round(stat.get('like', 0) / view, 4)
            video_info['coin_rate'] = round(stat.get('coin', 0) / view, 4)
            video_info['favorite_rate'] = round(stat.get('favorite', 0) / view, 4)
            video_info['share_rate'] = round(stat.get('share', 0) / view, 4)
            video_info['danmaku_rate'] = round(stat.get('danmaku', 0) / view, 4)
            video_info['reply_rate'] = round(stat.get('reply', 0) / view, 4)
        else:
            for key in ['like_rate', 'coin_rate', 'favorite_rate', 'share_rate', 'danmaku_rate', 'reply_rate']:
                video_info[key] = 0

        return video_info

    @staticmethod
    def merge_user_info(videos: List[Dict], user_info_dict: Dict[int, Dict]) -> List[Dict]:
        for video in videos:
            mid = video['mid']
            if mid in user_info_dict:
                info = user_info_dict[mid]
                video['owner_level'] = info.get('level')
                video['owner_fans'] = info.get('fans')
                video['owner_vip_type'] = info.get('vip_type')
                video['owner_official_verify'] = info.get('official_verify')
        return videos

    @staticmethod
    def deduplicate_videos(videos: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for video in videos:
            bvid = video['bvid']
            if bvid not in seen:
                seen.add(bvid)
                unique.append(video)
        removed = len(videos) - len(unique)
        logger.info(f"去重完成: 原始{len(videos)}条，去重{removed}条，剩余{len(unique)}条")
        return unique

    @staticmethod
    def to_dataframe(videos: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(videos)
        df = df.sort_values('view', ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        df['pubdate_dt'] = pd.to_datetime(df['pubdate'], unit='s')
        df['pub_hour'] = df['pubdate_dt'].dt.hour
        df['pub_dayofweek'] = df['pubdate_dt'].dt.dayofweek
        df['pub_day'] = df['pubdate_dt'].dt.day
        df['pub_month'] = df['pubdate_dt'].dt.month
        df['pub_year'] = df['pubdate_dt'].dt.year
        df['is_weekend'] = df['pub_dayofweek'].isin([5, 6]).astype(int)

        def cat_duration(s):
            if s < 180: return '短视频'
            elif s < 600: return '中等'
            elif s < 1800: return '长视频'
            else: return '超长视频'

        def cat_level(level):
            if pd.isna(level): return '未知'
            elif level <= 2: return '新手'
            elif level <= 4: return '普通'
            elif level <= 5: return '高级'
            else: return '大佬'

        def cat_fans(fans):
            if pd.isna(fans): return '未知'
            elif fans < 10000: return '小UP'
            elif fans < 100000: return '中UP'
            elif fans < 1000000: return '大UP'
            else: return '头部UP'

        df['duration_category'] = df['duration'].apply(cat_duration)
        df['owner_level_category'] = df['owner_level'].apply(cat_level)
        df['owner_fans_category'] = df['owner_fans'].apply(cat_fans)

        for col in ['view', 'like', 'coin', 'favorite', 'share', 'danmaku', 'reply']:
            max_val = df[col].max()
            df[f'{col}_normalized'] = df[col] / max_val if max_val > 0 else 0

        df['popularity_score'] = (
            df['view_normalized'] * 0.3 +
            df['like_normalized'] * 0.2 +
            df['coin_normalized'] * 0.2 +
            df['favorite_normalized'] * 0.15 +
            df['share_normalized'] * 0.05 +
            df['danmaku_normalized'] * 0.05 +
            df['reply_normalized'] * 0.05
        ) * 100

        return df
