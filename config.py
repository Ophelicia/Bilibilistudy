"""
B站爬虫配置文件
"""

# ==================== Cookie配置 ====================
COOKIE = "buvid3=536883B0-273F-8644-6CFF-4BFDFB19EC3190330infoc; b_nut=1766118390; _uuid=E126CCE10-63106-9894-4EBD-CB1E1FE64921089648infoc; buvid4=44C0F1F3-5B73-CC5B-752C-FC689DBE65AC81193-025071308-j0VA6HQPwStwN8IkmE+ufw%3D%3D; buvid_fp=2c5dd62e131cae098efc52980a207747; DedeUserID=3582258; DedeUserID__ckMd5=4eca37f701ce6817; theme-tip-show=SHOWED; theme-avatar-tip-show=SHOWED; CURRENT_QUALITY=116; rpdid=|(k~RR|u)Ymm0J'u~YYuJ~~u~; LIVE_BUVID=AUTO5017683166457713; hit-dyn-v2=1; CURRENT_LANGUAGE=; theme_style=light; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzIzODI0MDcsImlhdCI6MTc3MjEyMzE0NywicGx0IjotMX0.4BSBnD-M6PXnzqezBhcWsn_oYH0Ged0Z7E5Yl-giXJQ; bili_ticket_expires=1772382347; SESSDATA=b69aef05%2C1787842637%2C2431e%2A22CjDDZ3XrX6XT13bsAHqh-RnKocaGhrwJ0EAclZxaB4KqxwDeZXRb-1yh-QrV0d9JMB0SVkZ3eGdmTjBfVGIyekJSOGg1Ri1ZT2tQR0p4cWV0RHZVWjkxcXRQTjlEcW5zdjJSckY3eUM3TkhpSVJLSkJtREFLaVFyZEJaVnc1aUdaRHVCYl84R21nIIEC; bili_jct=1d6ef0cd2b8214e74524a1c9b8a501ce; sid=8gvvc17f; bsource=search_google; PVID=4; CURRENT_FNVAL=4048; bp_t_offset_3582258=1174683562729799680; home_feed_column=4; browser_resolution=1310-911; b_lsid=DA9429BF_19CA7DCA874"

# ==================== 请求头配置 ====================
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com',
    'Origin': 'https://www.bilibili.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'Cookie': COOKIE
}

# ==================== API配置 ====================
POPULAR_API = "https://api.bilibili.com/x/web-interface/popular"
RANKING_API = "https://api.bilibili.com/x/web-interface/ranking/v2"
USER_INFO_API = "https://api.bilibili.com/x/space/wbi/acc/info"
NAV_API = "https://api.bilibili.com/x/web-interface/nav"

# ==================== 分区配置（排除影视类） ====================
REGION_IDS = {
    1: "动画",
    3: "音乐",
    4: "游戏",
    5: "娱乐",
    36: "知识",
    188: "科技",
    160: "生活",
    211: "美食",
    217: "动物圈",
    119: "鬼畜",
    155: "时尚",
}

# ==================== 爬取配置 ====================
POPULAR_PAGES = 5
REGION_VIDEO_COUNT = 100
TARGET_MIN = 1000
TARGET_MAX = 2000

# ==================== 反爬配置 ====================
REQUEST_DELAY_MIN = 1
REQUEST_DELAY_MAX = 3
MAX_RETRIES = 3
TIMEOUT = 10

# ==================== 数据存储配置 ====================
OUTPUT_CSV = "bilibili_videos_data.csv"
OUTPUT_JSON = "bilibili_videos_data.json"
OUTPUT_EXCEL = "bilibili_videos_data.xlsx"
SAVE_INTERMEDIATE = True
INTERMEDIATE_FILE = "bilibili_videos_intermediate.json"
