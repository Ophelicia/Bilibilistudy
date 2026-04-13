"""
B站视频数据爬虫主程序
"""

import json
from typing import List, Dict
from bilibili_api import BilibiliAPI
from data_processor import DataProcessor
from utils import logger
from config import (
    POPULAR_PAGES, REGION_IDS, REGION_VIDEO_COUNT,
    TARGET_MIN, TARGET_MAX, SAVE_INTERMEDIATE, INTERMEDIATE_FILE
)


class BilibiliSpider:

    def __init__(self):
        self.api = BilibiliAPI()
        self.processor = DataProcessor()
        self.all_videos = []

    def crawl_popular_videos(self) -> List[Dict]:
        logger.info("=" * 50)
        logger.info("开始爬取综合热门视频")
        popular_videos = []
        for page in range(1, POPULAR_PAGES + 1):
            videos_raw = self.api.get_popular_videos(page=page, page_size=20)
            for v in videos_raw:
                popular_videos.append(self.processor.extract_video_info(v, source='popular'))
        logger.info(f"综合热门爬取完成: 共{len(popular_videos)}条")
        return popular_videos

    def crawl_ranking_videos(self) -> List[Dict]:
        logger.info("=" * 50)
        logger.info("开始爬取分区热门视频")
        ranking_videos = []
        for rid, name in REGION_IDS.items():
            logger.info(f"正在爬取【{name}】分区...")
            videos_raw = self.api.get_ranking_videos(rid=rid, page_size=REGION_VIDEO_COUNT)
            for v in videos_raw:
                ranking_videos.append(self.processor.extract_video_info(v, source=f'ranking_{name}'))
        logger.info(f"分区热门爬取完成: 共{len(ranking_videos)}条")
        return ranking_videos

    def crawl_all_videos(self) -> List[Dict]:
        popular = self.crawl_popular_videos()
        ranking = self.crawl_ranking_videos()
        all_videos = popular + ranking
        logger.info(f"数据合并完成: 总计{len(all_videos)}条原始数据")
        all_videos = self.processor.deduplicate_videos(all_videos)

        if len(all_videos) < TARGET_MIN:
            logger.warning(f"数据量({len(all_videos)})低于目标({TARGET_MIN})")
        elif len(all_videos) > TARGET_MAX:
            all_videos = sorted(all_videos, key=lambda x: x['view'], reverse=True)[:TARGET_MAX]

        self.all_videos = all_videos
        if SAVE_INTERMEDIATE:
            self._save_intermediate(all_videos)
        return all_videos

    def enrich_with_user_info(self, videos: List[Dict], batch_size: int = 50) -> List[Dict]:
        logger.info("开始获取UP主详细信息")
        unique_mids = list(set(v['mid'] for v in videos))
        logger.info(f"需要获取{len(unique_mids)}个UP主的信息")
        user_info_dict = self.api.batch_get_user_info(unique_mids, batch_size=batch_size)
        videos = self.processor.merge_user_info(videos, user_info_dict)
        if SAVE_INTERMEDIATE:
            self._save_intermediate(videos)
        return videos

    def _save_intermediate(self, videos: List[Dict]):
        try:
            with open(INTERMEDIATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(videos, f, ensure_ascii=False, indent=2)
            logger.info(f"中间数据已保存: {INTERMEDIATE_FILE}")
        except Exception as e:
            logger.error(f"保存中间数据失败: {e}")

    def load_intermediate(self) -> List[Dict]:
        try:
            with open(INTERMEDIATE_FILE, 'r', encoding='utf-8') as f:
                videos = json.load(f)
            logger.info(f"成功加载中间数据: {len(videos)}条")
            return videos
        except FileNotFoundError:
            logger.warning(f"中间数据文件不存在: {INTERMEDIATE_FILE}")
            return []
        except Exception as e:
            logger.error(f"加载中间数据失败: {e}")
            return []
