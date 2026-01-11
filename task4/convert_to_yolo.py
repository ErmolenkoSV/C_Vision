"""
Скрипт для конвертации аннотаций из CSV формата в формат YOLO
"""
import pandas as pd
import os
from pathlib import Path
from collections import OrderedDict

def convert_csv_to_yolo(csv_path, images_dir, output_dir):
    """Конвертирует аннотации из CSV в формат YOLO"""
    df = pd.read_csv(csv_path)
    
    unique_classes = sorted(df['class'].unique())
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    print(f"Классов: {len(unique_classes)}")
    print(f"Классы: {unique_classes}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = output_dir / "labels"
    images_output_dir = output_dir / "images"
    labels_dir.mkdir(exist_ok=True)
    images_output_dir.mkdir(exist_ok=True)
    
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        img_width = group.iloc[0]['width']
        img_height = group.iloc[0]['height']
        
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                
                center_x = ((xmin + xmax) / 2.0) / img_width
                center_y = ((ymin + ymax) / 2.0) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                class_id = class_to_id[row['class']]
                
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        src_image = Path(images_dir) / filename
        dst_image = images_output_dir / filename
        if src_image.exists():
            import shutil
            shutil.copy2(src_image, dst_image)
    
    classes_file = output_dir / "classes.txt"
    with open(classes_file, 'w') as f:
        for cls in unique_classes:
            f.write(f"{cls}\n")
    
    mapping_file = output_dir / "class_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("class_id,class_name\n")
        for cls, idx in class_to_id.items():
            f.write(f"{idx},{cls}\n")
    
    print(f"\nЗавершено!")
    print(f"Изображений: {len(grouped)}")
    print(f"Папка: {output_dir}")
    print(f"Классы: {classes_file}")
    
    return class_to_id, unique_classes

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    archive_dir = script_dir / "archive"
    
    csv_path = archive_dir / "annotations.csv"
    images_dir = archive_dir / "images"
    output_dir = script_dir / "yolo_dataset"
    
    if not csv_path.exists():
        print(f"Ошибка: файл {csv_path} не найден!")
        exit(1)
    
    if not images_dir.exists():
        print(f"Ошибка: папка {images_dir} не найдена!")
        exit(1)
    
    convert_csv_to_yolo(csv_path, images_dir, output_dir)


