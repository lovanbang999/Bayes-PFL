from multiprocessing import Pool, cpu_count
import os
import json
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import time

class MapReduceDataProcessor:
    """
    MapReduce Framework cho xử lý dataset anomaly detection
    
    Workflow:
    1. MAP: Xử lý từng ảnh song song (load, resize, augment, extract stats)
    2. SHUFFLE & SORT: Nhóm kết quả theo class_name
    3. REDUCE: Tổng hợp statistics cho mỗi class
    """
    
    def __init__(self, num_workers=None):
        """
        Args:
            num_workers: Số lượng worker processes (default: số CPU cores)
        """
        self.num_workers = num_workers or cpu_count()
        print(f"Initializing MapReduce with {self.num_workers} workers")
    
    # ==================== MAP PHASE ====================
    def map_process_image(self, img_info):
        """
        MAP Function: Xử lý một ảnh đơn lẻ
        
        Input: img_info dict chứa thông tin ảnh
        Output: (key, value) tuple
            - key: class_name (để group trong shuffle phase)
            - value: dictionary chứa statistics và metadata
        """
        try:
            img_path = img_info['img_path']
            mask_path = img_info['mask_path']
            cls_name = img_info['cls_name']
            anomaly = img_info['anomaly']
            
            # Load image
            if not os.path.exists(img_path):
                return (None, None)
            
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Tính toán statistics cơ bản
            stats = {
                'img_path': img_path,
                'cls_name': cls_name,
                'anomaly': anomaly,
                'shape': img_array.shape,
                'mean_rgb': img_array.mean(axis=(0, 1)).tolist(),
                'std_rgb': img_array.std(axis=(0, 1)).tolist(),
                'min_rgb': img_array.min(axis=(0, 1)).tolist(),
                'max_rgb': img_array.max(axis=(0, 1)).tolist(),
                'file_size': os.path.getsize(img_path),
            }
            
            # Nếu là ảnh anomaly, xử lý mask
            if anomaly == 1 and mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask_array = np.array(mask)
                stats['mask_coverage'] = (mask_array > 0).sum() / mask_array.size
                stats['has_mask'] = True
            else:
                stats['mask_coverage'] = 0.0
                stats['has_mask'] = False
            
            # Key: class_name để group, Value: statistics
            return (cls_name, stats)
            
        except Exception as e:
            print(f"Error processing {img_info.get('img_path', 'unknown')}: {e}")
            return (None, None)
    
    def map_validate_image(self, img_info):
        """
        MAP Function: Kiểm tra tính hợp lệ của ảnh
        
        Output: (key, value)
            - key: 'valid' hoặc 'invalid'
            - value: img_info
        """
        try:
            img_path = img_info['img_path']
            
            # Kiểm tra file tồn tại
            if not os.path.exists(img_path):
                return ('invalid', {'reason': 'file_not_found', 'path': img_path})
            
            # Kiểm tra load được không
            img = Image.open(img_path)
            img.verify()  # Verify image integrity
            
            # Kiểm tra kích thước
            img = Image.open(img_path)  # Reopen after verify
            width, height = img.size
            
            if width < 50 or height < 50:
                return ('invalid', {'reason': 'too_small', 'path': img_path, 'size': (width, height)})
            
            return ('valid', img_info)
            
        except Exception as e:
            return ('invalid', {'reason': str(e), 'path': img_info.get('img_path', 'unknown')})
    
    # ==================== SHUFFLE & SORT PHASE ====================
    def shuffle_and_sort(self, mapped_data):
        """
        Nhóm dữ liệu theo key (class_name)
        
        Input: List of (key, value) tuples
        Output: Dictionary {key: [value1, value2, ...]}
        """
        print("Shuffling and sorting data...")
        grouped = {}
        
        for key, value in mapped_data:
            if key is None or value is None:
                continue
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(value)
        
        print(f"Grouped data into {len(grouped)} classes")
        return grouped
    
    # ==================== REDUCE PHASE ====================
    def reduce_class_statistics(self, cls_name, stats_list):
        """
        REDUCE Function: Tổng hợp statistics cho một class
        
        Input: 
            - cls_name: tên class
            - stats_list: list of statistics từ MAP phase
        Output: Aggregated statistics dictionary
        """
        if not stats_list:
            return None
        
        # Tính toán aggregated statistics
        total_images = len(stats_list)
        normal_images = sum(1 for s in stats_list if s['anomaly'] == 0)
        anomaly_images = sum(1 for s in stats_list if s['anomaly'] == 1)
        
        # Mean RGB across all images
        all_means = np.array([s['mean_rgb'] for s in stats_list])
        all_stds = np.array([s['std_rgb'] for s in stats_list])
        
        # File sizes
        total_size = sum(s['file_size'] for s in stats_list)
        
        # Mask coverage (for anomaly images)
        anomaly_stats = [s for s in stats_list if s['anomaly'] == 1]
        avg_mask_coverage = np.mean([s['mask_coverage'] for s in anomaly_stats]) if anomaly_stats else 0.0
        
        # Image dimensions
        shapes = [s['shape'] for s in stats_list]
        
        result = {
            'cls_name': cls_name,
            'total_images': total_images,
            'normal_images': normal_images,
            'anomaly_images': anomaly_images,
            'anomaly_ratio': anomaly_images / total_images if total_images > 0 else 0,
            'mean_rgb': all_means.mean(axis=0).tolist(),
            'std_rgb': all_stds.mean(axis=0).tolist(),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_mb': (total_size / total_images) / (1024 * 1024) if total_images > 0 else 0,
            'avg_mask_coverage': float(avg_mask_coverage),
            'image_dimensions': {
                'min': [min(s[i] for s in shapes) for i in range(2)],
                'max': [max(s[i] for s in shapes) for i in range(2)],
                'avg': [np.mean([s[i] for s in shapes]) for i in range(2)]
            }
        }
        
        return result
    
    def reduce_validation_results(self, validation_grouped):
        """
        REDUCE Function: Tổng hợp kết quả validation
        """
        valid_images = validation_grouped.get('valid', [])
        invalid_images = validation_grouped.get('invalid', [])
        
        # Phân loại lý do invalid
        invalid_reasons = {}
        for item in invalid_images:
            reason = item.get('reason', 'unknown')
            if reason not in invalid_reasons:
                invalid_reasons[reason] = []
            invalid_reasons[reason].append(item['path'])
        
        result = {
            'total_valid': len(valid_images),
            'total_invalid': len(invalid_images),
            'invalid_reasons': {k: len(v) for k, v in invalid_reasons.items()},
            'invalid_details': invalid_reasons
        }
        
        return result
    
    # ==================== MAIN PIPELINE ====================
    def run_mapreduce_statistics(self, dataset_items):
        """
        Chạy MapReduce để tính statistics
        
        Args:
            dataset_items: List of image info dictionaries
        
        Returns:
            Dictionary chứa statistics cho mỗi class
        """
        print(f"\n{'='*60}")
        print(f"Starting MapReduce Statistics Pipeline")
        print(f"Total items: {len(dataset_items)}")
        print(f"Workers: {self.num_workers}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # ===== MAP PHASE =====
        print("📊 MAP PHASE: Processing images in parallel...")
        with Pool(self.num_workers) as pool:
            mapped_results = list(tqdm(
                pool.imap(self.map_process_image, dataset_items),
                total=len(dataset_items),
                desc="Mapping"
            ))
        
        map_time = time.time() - start_time
        print(f"✓ MAP completed in {map_time:.2f}s")
        
        # ===== SHUFFLE & SORT PHASE =====
        shuffle_start = time.time()
        grouped_data = self.shuffle_and_sort(mapped_results)
        shuffle_time = time.time() - shuffle_start
        print(f"✓ SHUFFLE & SORT completed in {shuffle_time:.2f}s")
        
        # ===== REDUCE PHASE =====
        print("\n📈 REDUCE PHASE: Aggregating statistics...")
        reduce_start = time.time()
        reduced_results = {}
        
        for cls_name, stats_list in tqdm(grouped_data.items(), desc="Reducing"):
            reduced_results[cls_name] = self.reduce_class_statistics(
                cls_name, stats_list
            )
        
        reduce_time = time.time() - reduce_start
        print(f"✓ REDUCE completed in {reduce_time:.2f}s")
        
        # ===== SUMMARY =====
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"MapReduce Pipeline Completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"  - MAP: {map_time:.2f}s ({map_time/total_time*100:.1f}%)")
        print(f"  - SHUFFLE: {shuffle_time:.2f}s ({shuffle_time/total_time*100:.1f}%)")
        print(f"  - REDUCE: {reduce_time:.2f}s ({reduce_time/total_time*100:.1f}%)")
        print(f"Classes processed: {len(reduced_results)}")
        print(f"{'='*60}\n")
        
        return reduced_results
    
    def run_mapreduce_validation(self, dataset_items):
        """
        Chạy MapReduce để validate dataset
        """
        print(f"\n{'='*60}")
        print(f"Starting MapReduce Validation Pipeline")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # MAP PHASE
        print("🔍 MAP PHASE: Validating images...")
        with Pool(self.num_workers) as pool:
            mapped_results = list(tqdm(
                pool.imap(self.map_validate_image, dataset_items),
                total=len(dataset_items),
                desc="Validating"
            ))
        
        # SHUFFLE & SORT
        grouped_data = self.shuffle_and_sort(mapped_results)
        
        # REDUCE
        print("\n📋 REDUCE PHASE: Aggregating validation results...")
        validation_summary = self.reduce_validation_results(grouped_data)
        
        total_time = time.time() - start_time
        print(f"\n✓ Validation completed in {total_time:.2f}s")
        
        return validation_summary, grouped_data.get('valid', [])


# ==================== HELPER FUNCTIONS ====================
def load_dataset_metadata(meta_path):
    """Load metadata từ JSON file"""
    with open(meta_path, 'r') as f:
        return json.load(f)


def prepare_dataset_items(dataset_root, meta_path):
    """
    Chuẩn bị danh sách items từ metadata (supports both flat and nested format)
    """
    meta_info = load_dataset_metadata(meta_path)
    all_items = []
    
    # Check format: nested (train/test) or flat (numeric keys)
    if 'train' in meta_info or 'test' in meta_info:
        # Nested format
        for phase in ['train', 'test']:
            if phase not in meta_info:
                continue
            
            for cls_name, items in meta_info[phase].items():
                for item in items:
                    full_item = item.copy()
                    full_item['img_path'] = os.path.join(dataset_root, item['img_path'])
                    if item['mask_path']:
                        full_item['mask_path'] = os.path.join(dataset_root, item['mask_path'])
                    full_item['phase'] = phase
                    all_items.append(full_item)
    else:
        # Flat format - keys are numeric strings
        for key, item in meta_info.items():
            if isinstance(item, dict):
                # Convert flat format keys to expected format
                full_item = {
                    'img_path': item.get('image_path', item.get('img_path', '')),
                    'mask_path': item.get('mask_path', ''),
                    'cls_name': item.get('clsname', item.get('cls_name', '')),
                    'anomaly': item.get('label', item.get('anomaly', 0)),
                    'specie_name': item.get('anomaly', 'unknown'),
                    'phase': item.get('split', 'unknown')
                }
                all_items.append(full_item)
    
    return all_items

def save_statistics(stats, output_path):
    """Lưu statistics ra file JSON"""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"✓ Statistics saved to: {output_path}")


def print_statistics_summary(stats):
    """In tóm tắt statistics"""
    print(f"\n{'='*60}")
    print("DATASET STATISTICS SUMMARY")
    print(f"{'='*60}")
    
    total_images = sum(s['total_images'] for s in stats.values())
    total_size_mb = sum(s['total_size_mb'] for s in stats.values())
    
    print(f"Total classes: {len(stats)}")
    print(f"Total images: {total_images}")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"\nPer-class breakdown:")
    print(f"{'Class':<20} {'Total':<8} {'Normal':<8} {'Anomaly':<8} {'Size(MB)':<10}")
    print(f"{'-'*60}")
    
    for cls_name, cls_stats in sorted(stats.items()):
        print(f"{cls_name:<20} {cls_stats['total_images']:<8} "
              f"{cls_stats['normal_images']:<8} {cls_stats['anomaly_images']:<8} "
              f"{cls_stats['total_size_mb']:<10.2f}")
    
    print(f"{'='*60}\n")


# ==================== MAIN EXECUTION ====================
def process_single_dataset(dataset_name, dataset_root, meta_path, num_workers=None):
    """
    Xử lý một dataset với MapReduce
    
    Args:
        dataset_name: Tên dataset (e.g., 'visa', 'mvtec')
        dataset_root: Đường dẫn root của dataset
        meta_path: Đường dẫn file metadata JSON
        num_workers: Số workers (None = auto)
    
    Returns:
        Dictionary chứa statistics
    """
    print(f"\n🚀 Processing dataset: {dataset_name}")
    print(f"Root: {dataset_root}")
    print(f"Meta: {meta_path}")
    
    # Load items
    print("\n📁 Loading dataset metadata...")
    all_items = prepare_dataset_items(dataset_root, meta_path)
    print(f"✓ Loaded {len(all_items)} items")
    
    # Initialize MapReduce processor
    mr_processor = MapReduceDataProcessor(num_workers=num_workers)
    
    # Run validation (optional)
    print("\n" + "="*60)
    validation_summary, valid_items = mr_processor.run_mapreduce_validation(all_items)
    print(f"\nValidation Results:")
    print(f"  Valid: {validation_summary['total_valid']}")
    print(f"  Invalid: {validation_summary['total_invalid']}")
    if validation_summary['invalid_reasons']:
        print(f"  Invalid reasons: {validation_summary['invalid_reasons']}")
    
    # Run statistics computation
    class_statistics = mr_processor.run_mapreduce_statistics(valid_items)
    
    # Print summary
    print_statistics_summary(class_statistics)
    
    # Save results
    output_path = meta_path.replace('.json', '_mapreduce_stats.json')
    save_statistics(class_statistics, output_path)
    
    return class_statistics


if __name__ == '__main__':
    """
    Example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MapReduce Data Processing')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset name (visa, mvtec, etc.)')
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root directory')
    parser.add_argument('--meta', type=str, required=True,
                        help='Metadata JSON file path')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: CPU count)')
    
    args = parser.parse_args()
    
    # Process dataset
    stats = process_single_dataset(
        dataset_name=args.dataset,
        dataset_root=args.root,
        meta_path=args.meta,
        num_workers=args.workers
    )
    