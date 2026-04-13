"""
独立的失败重试脚本
可以单独运行此脚本来重试失败的UP主信息
"""

from bilibili_api import BilibiliAPI
from data_processor import DataProcessor
from utils import logger
from config import COOKIE
import json
import pandas as pd


def main():
    logger.info("\n" + "🔄" * 25)
    logger.info("B站UP主信息失败重试工具")
    logger.info("🔄" * 25 + "\n")
    
    # 检查Cookie
    if not COOKIE or COOKIE == "your_cookie_here":
        logger.error("❌ 请先在config.py中配置您的B站Cookie！")
        return
    
    # 初始化API
    api = BilibiliAPI()
    processor = DataProcessor()
    
    # 检查失败文件是否存在
    try:
        with open('failed_mids.json', 'r', encoding='utf-8') as f:
            failed_data = json.load(f)
            failed_mids = failed_data.get('failed_mids', [])
    except FileNotFoundError:
        logger.error("❌ 未找到 failed_mids.json 文件")
        logger.error("请先运行主程序进行爬取")
        return
    
    if not failed_mids:
        logger.info("✅ 没有失败的UP主信息需要重试")
        return
    
    logger.info(f"找到{len(failed_mids)}个失败的UP主")
    
    # 询问重试策略
    print("\n请选择重试策略：")
    print("1. 保守模式（每20个休息，延时长）")
    print("2. 平衡模式（每50个休息）")
    print("3. 仅重试前N个")
    
    choice = input("请选择 (1/2/3，默认1): ").strip() or "1"
    
    if choice == "3":
        n = int(input("请输入要重试的数量: ").strip())
        failed_mids = failed_mids[:n]
        logger.info(f"将重试前{n}个UP主")
    
    batch_size = 20 if choice == "1" else 50
    
    # 开始重试
    logger.info(f"\n开始重试，批次大小={batch_size}")
    user_info_dict = api.batch_get_user_info(failed_mids, batch_size=batch_size)
    
    # 统计结果
    success_count = len(user_info_dict)
    still_failed = len(api.failed_mids)
    
    logger.info(f"\n{'='*50}")
    logger.info("重试结果:")
    logger.info(f"  - 重试数量: {len(failed_mids)}")
    logger.info(f"  - 成功: {success_count}")
    logger.info(f"  - 仍然失败: {still_failed}")
    logger.info(f"  - 成功率: {success_count/len(failed_mids)*100:.1f}%")
    logger.info(f"{'='*50}\n")
    
    if success_count == 0:
        logger.warning("❌ 本次重试未获取到任何新数据")
        return
    
    # 更新原始数据
    try:
        with open('bilibili_videos_intermediate.json', 'r', encoding='utf-8') as f:
            videos = json.load(f)
        
        logger.info(f"加载了{len(videos)}条视频数据")
        
        # 合并UP主信息
        videos = processor.merge_user_info(videos, user_info_dict)
        
        # 保存更新后的数据
        with open('bilibili_videos_intermediate.json', 'w', encoding='utf-8') as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 中间数据已更新")
        
        # 重新生成最终文件
        df = processor.to_dataframe(videos)
        df = processor.add_derived_features(df)
        
        df.to_csv('bilibili_videos_data.csv', index=False, encoding='utf-8-sig')
        df.to_json('bilibili_videos_data.json', orient='records', force_ascii=False, indent=2)
        df.to_excel('bilibili_videos_data.xlsx', index=False, engine='openpyxl')
        
        logger.info("✅ 所有输出文件已更新")
        
        # 统计UP主信息完整性
        complete_count = df['owner_level'].notna().sum()
        logger.info(f"\nUP主信息完整性: {complete_count}/{len(df)} ({complete_count/len(df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ 更新数据时出错: {e}")
    
    logger.info("\n" + "✅" * 25)
    logger.info("重试完成！")
    logger.info("✅" * 25 + "\n")


if __name__ == "__main__":
    main()
