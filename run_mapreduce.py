#!/usr/bin/env python3
"""
Standalone script để chạy MapReduce trên datasets
"""

from mapreduce_dataset import process_single_dataset
import os

def main():
    # Danh sách datasets cần xử lý
    datasets = [
        {
            'name': 'visa',
            'root': './dataset/mvisa/data/visa',
            'meta': './dataset/mvisa/data/meta_visa.json'
        },
        {
            'name': 'mvtec',
            'root': './dataset/mvisa/data/mvtec',
            'meta': './dataset/mvisa/data/meta_mvtec.json'
        },
        {
            'name': 'your_custom_dataset',  # Dataset tự thu
            'root': './dataset/mvisa/data/your_custom_dataset',
            'meta': './dataset/mvisa/data/meta_your_custom_dataset.json'
        }
    ]
    
    print("="*80)
    print("BATCH MAPREDUCE PROCESSING FOR ALL DATASETS")
    print("="*80)
    
    all_stats = {}
    
    for dataset_config in datasets:
        if not os.path.exists(dataset_config['meta']):
            print(f"\n⚠️  Skipping {dataset_config['name']}: metadata not found")
            continue
        
        stats = process_single_dataset(
            dataset_name=dataset_config['name'],
            dataset_root=dataset_config['root'],
            meta_path=dataset_config['meta'],
            num_workers=8
        )
        
        all_stats[dataset_config['name']] = stats
    
    # Tổng hợp kết quả
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_images = 0
    total_size_gb = 0
    
    for dataset_name, stats in all_stats.items():
        dataset_images = sum(s['total_images'] for s in stats.values())
        dataset_size_gb = sum(s['total_size_mb'] for s in stats.values()) / 1024
        
        total_images += dataset_images
        total_size_gb += dataset_size_gb
        
        print(f"{dataset_name}:")
        print(f"  Images: {dataset_images}")
        print(f"  Size: {dataset_size_gb:.2f} GB")
        print(f"  Classes: {len(stats)}")
    
    print(f"\nTotal across all datasets:")
    print(f"  Images: {total_images}")
    print(f"  Size: {total_size_gb:.2f} GB")
    print(f"  Datasets: {len(all_stats)}")
    
    # Kiểm tra yêu cầu
    print(f"\n{'='*80}")
    print("REQUIREMENT CHECK:")
    for dataset_name, stats in all_stats.items():
        size_gb = sum(s['total_size_mb'] for s in stats.values()) / 1024
        status = "✓ PASS" if size_gb >= 10 else "✗ FAIL"
        print(f"  {dataset_name}: {size_gb:.2f} GB {status}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
