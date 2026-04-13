"""
一键运行所有数据分析
"""

import sys
from pathlib import Path

# 添加analysis目录到路径
analysis_path = Path(__file__).parent / 'analysis'
if str(analysis_path) not in sys.path:
    sys.path.insert(0, str(analysis_path))


def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")


def run_analysis_step(step_num, total_steps, title, module_name, function_name='main'):
    """运行单个分析步骤"""
    print_banner(f"步骤 {step_num}/{total_steps}: {title}")
    
    try:
        # 动态导入模块
        module = __import__(module_name)
        main_func = getattr(module, function_name)
        main_func()
        print(f"✅ {title} 完成\n")
        return True
    except Exception as e:
        print(f"❌ {title} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数 - 执行所有分析"""
    
    print_banner("B站视频选题成功关键因素研究 - 完整分析流程")
    
    print("本程序将依次执行以下分析:")
    print("  1. 描述性统计分析")
    print("  2. 聚类分析")
    print("  3. 单因素分析")
    print("  4. 回归分析")
    print("  5. 机器学习分析")
    print("  6. 分群对比分析")
    print("  7. 可视化报告汇总")
    print()
    
    # 检查数据文件是否存在
    data_file = Path('bilibili_videos_data.csv')
    if not data_file.exists():
        print("❌ 错误: 未找到数据文件 'bilibili_videos_data.csv'")
        print("请先运行爬虫程序获取数据")
        return
    
    choice = input("是否开始完整分析？(y/n，默认y): ").strip().lower() or 'y'
    
    if choice != 'y':
        print("已取消")
        return
    
    # 定义分析步骤
    steps = [
        (1, 7, "描述性统计分析", "1_descriptive_statistics"),
        (2, 7, "聚类分析", "2_clustering_analysis"),
        (3, 7, "单因素分析", "3_success_factors_univariate"),
        (4, 7, "回归分析", "4_success_factors_regression"),
        (5, 7, "机器学习分析", "5_success_factors_ml"),
        (6, 7, "分群对比分析", "6_group_comparison"),
        (7, 7, "可视化报告汇总", "7_visualization_report")
    ]
    
    # 记录成功和失败的步骤
    success_count = 0
    failed_steps = []
    
    # 执行所有步骤
    for step_num, total, title, module_name in steps:
        success = run_analysis_step(step_num, total, title, module_name)
        
        if success:
            success_count += 1
        else:
            failed_steps.append(title)
            
            # 询问是否继续
            if step_num < total:
                continue_choice = input(f"\n{title} 失败，是否继续下一步？(y/n，默认n): ").strip().lower()
                if continue_choice != 'y':
                    print("分析已中断")
                    break
    
    # 最终总结
    print_banner("分析完成总结")
    print(f"✅ 成功完成: {success_count}/{len(steps)} 个步骤")
    
    if failed_steps:
        print(f"❌ 失败步骤:")
        for step in failed_steps:
            print(f"   • {step}")
    else:
        print("🎉 所有分析步骤均成功完成！")
    
    print("\n分析结果保存在以下目录:")
    print("  • results/figures/  - 所有图表")
    print("  • results/tables/   - 所有数据表")
    print("  • results/reports/  - 分析报告")
    print()
    
    # 列出主要输出文件
    main_outputs = [
        ('results/reports/analysis_summary.txt', '分析总结'),
        ('results/reports/practical_recommendations.txt', '实践建议'),
        ('results/figures/interactive_dashboard.html', '交互式Dashboard'),
    ]
    
    print("主要输出文件:")
    for filepath, description in main_outputs:
        if Path(filepath).exists():
            print(f"  ✅ {description}: {filepath}")
        else:
            print(f"  ⚠️ {description}: {filepath} (未生成)")


if __name__ == "__main__":
    main()
