"""
One-click runner for all analysis scripts
"""

import subprocess
import sys
from pathlib import Path
import time


def print_banner(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")


def run_script(script_path, description):
    print(f"\n{'─' * 70}")
    print(f"  Running: {description}")
    print(f"{'─' * 70}\n")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True, capture_output=False, text=True
        )
        elapsed = time.time() - start
        print(f"\n✅ {description} - DONE ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError:
        elapsed = time.time() - start
        print(f"\n❌ {description} - FAILED ({elapsed:.1f}s)")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False


def main():
    print_banner("Bilibili Video Success Factor Analysis - Full Pipeline")

    data_file = Path('bilibili_videos_data.csv')
    if not data_file.exists():
        print("❌ Data file 'bilibili_videos_data.csv' not found!")
        print("   Please run main.py first to crawl data.")
        return

    print(f"📊 Data file found: {data_file}")
    print(f"   Size: {data_file.stat().st_size / 1024:.1f} KB")

    choice = input("\nStart analysis? (y/n, default y): ").strip().lower() or 'y'
    if choice != 'y':
        print("Cancelled.")
        return

    scripts = [
        (Path('analysis/1_descriptive_statistics.py'), '1. Descriptive Statistics'),
        (Path('analysis/2_clustering_analysis.py'), '2. Clustering Analysis'),
        (Path('analysis/3_success_factors_univariate.py'), '3. Univariate Factor Analysis'),
        (Path('analysis/4_success_factors_regression.py'), '4. Regression Analysis'),
        (Path('analysis/5_success_factors_ml.py'), '5. Machine Learning Analysis'),
        (Path('analysis/6_group_comparison.py'), '6. Group Comparison Analysis'),
        (Path('analysis/7_visualization_report.py'), '7. Report Generation'),
    ]

    # Check all scripts exist
    missing = [(p, d) for p, d in scripts if not p.exists()]
    if missing:
        print(f"\n❌ Missing scripts:")
        for p, d in missing:
            print(f"   - {p} ({d})")
        return

    print(f"\n📋 Pipeline: {len(scripts)} stages")
    for _, desc in scripts:
        print(f"   • {desc}")

    print_banner("Starting Analysis Pipeline")
    total_start = time.time()
    success_count = 0
    failed_list = []

    for i, (path, desc) in enumerate(scripts, 1):
        print(f"\n{'━' * 80}")
        print(f"  Stage {i}/{len(scripts)}: {desc}")
        print(f"{'━' * 80}")

        if run_script(path, desc):
            success_count += 1
        else:
            failed_list.append(desc)
            cont = input(f"\n  {desc} failed. Continue? (y/n, default y): ").strip().lower() or 'y'
            if cont != 'y':
                print("Pipeline stopped by user.")
                break

    total_time = time.time() - total_start

    # Final summary
    print_banner("Analysis Pipeline Complete")
    print(f"  ✅ Succeeded: {success_count}/{len(scripts)}")
    if failed_list:
        print(f"  ❌ Failed: {len(failed_list)}")
        for f in failed_list:
            print(f"     - {f}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")

    print(f"\n{'─' * 70}")
    print("  Output Directories:")
    print(f"{'─' * 70}")
    print("  📁 results/figures/  - Charts and visualizations")
    print("  📁 results/tables/   - Data tables (CSV)")
    print("  📁 results/reports/  - Analysis reports (JSON/TXT)")

    # List generated files
    for subdir in ['figures', 'tables', 'reports']:
        dir_path = Path('results') / subdir
        if dir_path.exists():
            files = list(dir_path.iterdir())
            if files:
                print(f"\n  📂 results/{subdir}/ ({len(files)} files):")
                for f in sorted(files)[:10]:
                    size = f.stat().st_size / 1024
                    print(f"     {f.name} ({size:.1f} KB)")
                if len(files) > 10:
                    print(f"     ... and {len(files) - 10} more files")

    print(f"\n{'═' * 80}")
    print("  🎉 Analysis pipeline finished!")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()
