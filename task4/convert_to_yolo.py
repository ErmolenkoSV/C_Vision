"""
Скрипт для конвертации аннотаций из CSV формата в формат YOLO
"""
import pandas as pd
import os
from pathlib import Path
from collections import OrderedDict

def convert_csv_to_yolo(csv_path, images_dir, output_dir):
    """
    Конвертирует аннотации из CSV в формат YOLO
    
    Args:
        csv_path: путь к файлу annotations.csv
        images_dir: путь к папке с изображениями
        output_dir: путь к выходной папке для YOLO формата
    """
    # Читаем CSV файл
    df = pd.read_csv(csv_path)
    
    # Получаем уникальные классы и создаем словарь
    unique_classes = sorted(df['class'].unique())
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    print(f"Найдено классов: {len(unique_classes)}")
    print(f"Классы: {unique_classes}")
    
    # Создаем структуру папок YOLO
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels_dir = output_dir / "labels"
    images_output_dir = output_dir / "images"
    labels_dir.mkdir(exist_ok=True)
    images_output_dir.mkdir(exist_ok=True)
    
    # Группируем аннотации по имени файла
    grouped = df.groupby('filename')
    
    # Создаем файлы аннотаций для каждого изображения
    for filename, group in grouped:
        # Получаем размеры изображения (должны быть одинаковыми для всех аннотаций одного файла)
        img_width = group.iloc[0]['width']
        img_height = group.iloc[0]['height']
        
        # Создаем файл аннотации
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for _, row in group.iterrows():
                # Конвертируем координаты из формата (xmin, ymin, xmax, ymax) в YOLO формат (center_x, center_y, width, height)
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                
                # Нормализуем координаты (от 0 до 1)
                center_x = ((xmin + xmax) / 2.0) / img_width
                center_y = ((ymin + ymax) / 2.0) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Получаем ID класса
                class_id = class_to_id[row['class']]
                
                # Записываем в формат YOLO: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        # Копируем изображение (или создаем символическую ссылку)
        src_image = Path(images_dir) / filename
        dst_image = images_output_dir / filename
        if src_image.exists():
            import shutil
            shutil.copy2(src_image, dst_image)
    
    # Сохраняем файл с классами
    classes_file = output_dir / "classes.txt"
    with open(classes_file, 'w') as f:
        for cls in unique_classes:
            f.write(f"{cls}\n")
    
    # Сохраняем mapping классов
    mapping_file = output_dir / "class_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("class_id,class_name\n")
        for cls, idx in class_to_id.items():
            f.write(f"{idx},{cls}\n")
    
    print(f"\nКонвертация завершена!")
    print(f"Всего изображений обработано: {len(grouped)}")
    print(f"Файлы сохранены в: {output_dir}")
    print(f"Классы сохранены в: {classes_file}")
    
    return class_to_id, unique_classes

if __name__ == "__main__":
    # Пути к файлам
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


