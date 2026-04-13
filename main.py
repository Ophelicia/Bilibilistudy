"""
B站视频数据爬虫 - 主程序入口
"""

import pandas as pd
import time
import random
from bilibili_spider import BilibiliSpider
from data_processor import DataProcessor
from utils import logger
from config import OUTPUT_CSV, OUTPUT_JSON, OUTPUT_EXCEL, COOKIE


def check_cookie():
    if not COOKIE or COOKIE == "your_cookie_here":
        logger.error("❌ 请先在config.py中配置您的B站Cookie！")
        return False
    return True


def main():
    logger.info("B站视频数据爬虫系统 v2.0")

    if not check_cookie():
        return

    spider = BilibiliSpider()
    processor = DataProcessor()

    print("\n请选择操作模式：")
    print("1. 完整爬取（从头开始）")
    print("2. 继续爬取（从中间数据继续）")
    print("3. 仅重试失败的UP主信息")
    print("4. 重新生成分析文件（基于现有数据）")

    mode = input("请选择 (1/2/3/4，默认1): ").strip() or "1"
    videos = None

    if mode == "3":
        failed_mids = spider.api.load_failed_mids()
        if failed_mids:
            logger.info(f"加载了{len(failed_mids)}个失败的mid")
            user_info_dict = spider.api.batch_get_user_info(failed_mids, batch_size=20)
            try:
                import json
                with open('bilibili_videos_intermediate.json', 'r', encoding='utf-8') as f:
                    videos = json.load(f)
                videos = processor.merge_user_info(videos, user_info_dict)
                with open('bilibili_videos_intermediate.json', 'w', encoding='utf-8') as f:
                    json.dump(videos, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"更新数据失败: {e}")
                return
        else:
            logger.info("没有需要重试的mid")
            return

    elif mode == "4":
        try:
            import json
            with open('bilibili_videos_intermediate.json', 'r', encoding='utf-8') as f:
                videos = json.load(f)
            logger.info(f"加载了{len(videos)}条视频数据")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return

    else:
        if mode == "2":
            videos = spider.load_intermediate()
            if not videos:
                videos = spider.crawl_all_videos()
        else:
            videos = spider.crawl_all_videos()

        if not videos:
            logger.error("❌ 未获取到任何视频数据")
            return

        print("\n是否获取UP主详细信息？")
        print("1. 是，批量获取（每50个休息一次）")
        print("2. 是，保守获取（每20个休息一次）")
        print("3. 否，跳过")

        choice = input("请选择 (1/2/3，默认3): ").strip() or "3"

        if choice == "1":
            videos = spider.enrich_with_user_info(videos, batch_size=50)
        elif choice == "2":
            videos = spider.enrich_with_user_info(videos, batch_size=20)

        stats = spider.api.get_statistics()
        if stats['failed_count'] > 0:
            logger.info(f"有{stats['failed_count']}个UP主信息获取失败，已保存到failed_mids.json")
            retry = input("是否立即重试？(y/n): ").strip().lower()
            if retry == 'y':
                time.sleep(30)
                retry_dict = spider.api.batch_get_user_info(
                    stats['failed_mids'], batch_size=20
                )
                videos = processor.merge_user_info(videos, retry_dict)

    if not videos:
        logger.error("❌ 没有可用的视频数据")
        return

    # 数据处理
    df = processor.to_dataframe(videos)
    df = processor.add_derived_features(df)

    # 统计
    logger.info(f"总视频数: {len(df)}")
    logger.info(f"涉及UP主数: {df['mid'].nunique()}")
    logger.info(f"涉及分区数: {df['tname'].nunique()}")
    logger.info(f"播放量 - 最小: {df['view'].min():,}, 最大: {df['view'].max():,}, 平均: {df['view'].mean():,.0f}")

    # 保存
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        logger.info(f"✅ CSV已保存: {OUTPUT_CSV}")
        df.to_json(OUTPUT_JSON, orient='records', force_ascii=False, indent=2)
        logger.info(f"✅ JSON已保存: {OUTPUT_JSON}")
        df.to_excel(OUTPUT_EXCEL, index=False, engine='openpyxl')
        logger.info(f"✅ Excel已保存: {OUTPUT_EXCEL}")
    except Exception as e:
        logger.error(f"保存文件时出错: {e}")

    logger.info("爬虫任务全部完成！")

    print("\n数据预览（前5条）:")
    cols = ['title', 'tname', 'view', 'like', 'owner_name']
    available = [c for c in cols if c in df.columns]
    print(df[available].head())


if __name__ == "__main__":
    main()
