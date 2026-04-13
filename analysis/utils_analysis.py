"""
Analysis utility functions with English translation support
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path('results')
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
REPORTS_DIR = RESULTS_DIR / 'reports'

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ==================== 中英文分区映射 ====================
REGION_CN_TO_EN = {
    # 游戏
    '手机游戏': 'Mobile Games',
    '单机游戏': 'PC Games',
    '网络游戏': 'Online Games',
    '电子竞技': 'Esports',
    '桌游棋牌': 'Board Games',
    '音游': 'Rhythm Games',
    'GMV': 'GMV',
    # 动画
    '动画综合': 'Animation',
    'MAD·AMV': 'MAD/AMV',
    'MMD·3D': 'MMD/3D',
    '短片': 'Short Films',
    '特摄': 'Tokusatsu',
    '动漫杂谈': 'Anime Talk',
    # 音乐
    '音乐综合': 'Music',
    '翻唱': 'Cover Songs',
    '演奏': 'Performance',
    '原创音乐': 'Original Music',
    'MV': 'MV',
    '乐评盘点': 'Music Review',
    '音乐现场': 'Live Music',
    '音乐教学': 'Music Tutorial',
    'AI音乐': 'AI Music',
    'VOCALOID·UTAU': 'VOCALOID/UTAU',
    '音MAD': 'OTOMAD',
    '电台': 'Radio',
    # 娱乐
    '娱乐杂谈': 'Entertainment Talk',
    '明星综合': 'Celebrity',
    '综艺': 'Variety Show',
    '粉丝创作': 'Fan Creation',
    # 知识
    '科学科普': 'Science',
    '社科·法律·心理': 'Social Science',
    '人文历史': 'History',
    '财经商业': 'Finance',
    '校园学习': 'Education',
    '职业职场': 'Career',
    # 科技
    '数码': 'Digital Tech',
    '电脑装机': 'PC Building',
    '手机平板': 'Phone/Tablet',
    '影音智能': 'Smart Devices',
    # 生活
    '搞笑': 'Comedy',
    '日常': 'Daily Life',
    '出行': 'Travel',
    '亲子': 'Parenting',
    '手工': 'Handcraft',
    '三农': 'Agriculture',
    '综合': 'General',
    # 美食
    '美食制作': 'Cooking',
    '美食记录': 'Food Vlog',
    '美食测评': 'Food Review',
    '田园美食': 'Rural Food',
    # 动物圈
    '动物圈': 'Animals',
    '喵星人': 'Cats',
    '汪星人': 'Dogs',
    '小宠异宠': 'Exotic Pets',
    '野生动物': 'Wildlife',
    '动物二创': 'Animal Remix',
    # 鬼畜
    '鬼畜调教': 'Remix',
    '鬼畜剧场': 'Parody',
    '人力VOCALOID': 'Human VOCALOID',
    # 时尚
    '美妆护肤': 'Beauty',
    '穿搭': 'Fashion',
    '仿妆cos': 'Cosplay',
    # 影视
    '影视杂谈': 'Film Talk',
    '影视剪辑': 'Film Editing',
    '影视综合': 'Film General',
    '预告·资讯': 'Trailers/News',
    # 运动
    '运动综合': 'Sports',
    '篮球': 'Basketball',
    '健身': 'Fitness',
    # 创作
    '摄影摄像': 'Photography',
    '绘画': 'Drawing',
    '设计·创意': 'Design',
    '野生技能协会': 'Skills',
    '模玩·周边': 'Toys/Merch',
    '颜值安利': 'Looks',
    # 汽车
    '汽车': 'Automobile',
    '新能源车': 'EV',
    # 其他
    '其他': 'Others',
    '同人·手书': 'Doujin/Sketch',
}

# 时长分类映射
DURATION_CN_TO_EN = {
    '短视频': 'Short (<3min)',
    '中等': 'Medium (3-10min)',
    '长视频': 'Long (10-30min)',
    '超长视频': 'Extra Long (>30min)',
}

# 星期映射
WEEKDAY_CN_TO_EN = {
    '周一': 'Mon', '周二': 'Tue', '周三': 'Wed',
    '周四': 'Thu', '周五': 'Fri', '周六': 'Sat', '周日': 'Sun',
}

# UP主等级映射
LEVEL_CN_TO_EN = {
    '新手': 'Beginner', '普通': 'Regular',
    '高级': 'Advanced', '大佬': 'Expert', '未知': 'Unknown',
}

# UP主粉丝映射
FANS_CN_TO_EN = {
    '小UP': 'Small UP', '中UP': 'Medium UP',
    '大UP': 'Large UP', '头部UP': 'Top UP', '未知': 'Unknown',
}


def translate_region(name):
    """Translate a single region name"""
    return REGION_CN_TO_EN.get(name, name)


def translate_dataframe(df):
    """Translate all Chinese labels in DataFrame to English"""
    df = df.copy()

    # Translate tname column
    if 'tname' in df.columns:
        df['tname_original'] = df['tname']
        df['tname'] = df['tname'].map(lambda x: REGION_CN_TO_EN.get(x, x))

    # Translate duration_category
    if 'duration_category' in df.columns:
        df['duration_category'] = df['duration_category'].map(
            lambda x: DURATION_CN_TO_EN.get(x, x)
        )

    # Translate owner_level_category
    if 'owner_level_category' in df.columns:
        df['owner_level_category'] = df['owner_level_category'].map(
            lambda x: LEVEL_CN_TO_EN.get(x, x)
        )

    # Translate owner_fans_category
    if 'owner_fans_category' in df.columns:
        df['owner_fans_category'] = df['owner_fans_category'].map(
            lambda x: FANS_CN_TO_EN.get(x, x)
        )

    return df


def load_data(file_path='bilibili_videos_data.csv', translate=True):
    """Load data with optional English translation"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    logger.info(f"✅ Data loaded: {len(df)} rows × {len(df.columns)} cols")
    if translate:
        df = translate_dataframe(df)
        logger.info("✅ Labels translated to English")
    return df


def save_figure(fig, filename, dpi=300):
    fig.savefig(FIGURES_DIR / filename, dpi=dpi, bbox_inches='tight')
    logger.info(f"✅ Figure saved: {FIGURES_DIR / filename}")
    plt.close(fig)


def save_table(df, filename):
    filepath = TABLES_DIR / filename
    if filename.endswith('.csv'):
        df.to_csv(filepath, index=True if isinstance(df.index, pd.MultiIndex) else False,
                  encoding='utf-8-sig')
    elif filename.endswith('.xlsx'):
        df.to_excel(filepath, index=False)
    logger.info(f"✅ Table saved: {filepath}")


def save_report(content, filename):
    filepath = REPORTS_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"✅ Report saved: {filepath}")


def calculate_success_index(df, view_w=0.4, engagement_w=0.4, completion_w=0.2):
    view_norm = (df['view'] - df['view'].min()) / (df['view'].max() - df['view'].min() + 1e-10)
    eng_norm = (df['engagement_rate'] - df['engagement_rate'].min()) / \
               (df['engagement_rate'].max() - df['engagement_rate'].min() + 1e-10)
    comp_norm = (df['completion_rate_proxy'] - df['completion_rate_proxy'].min()) / \
                (df['completion_rate_proxy'].max() - df['completion_rate_proxy'].min() + 1e-10)
    return (view_norm * view_w + eng_norm * engagement_w + comp_norm * completion_w) * 100


def get_basic_stats(series):
    return {
        'count': int(series.count()), 'mean': float(series.mean()),
        'std': float(series.std()), 'min': float(series.min()),
        'q25': float(series.quantile(0.25)), 'median': float(series.median()),
        'q75': float(series.quantile(0.75)), 'max': float(series.max()),
        'skew': float(series.skew()), 'kurtosis': float(series.kurtosis())
    }


def format_large_number(num):
    if num >= 1e8:
        return f"{num / 1e8:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.0f}"


def print_section_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")
