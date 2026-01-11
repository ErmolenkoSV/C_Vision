"""
Скрипт для обучения модели детектирования объектов с использованием Transfer Learning
Использует YOLOv8 с предобученными весами
"""
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil

def prepare_dataset(yolo_dataset_dir, train_split=0.8, val_split=0.1, test_split=0.1):
    """Разделяет датасет на train/val/test"""
    yolo_dataset_dir = Path(yolo_dataset_dir)
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6
    
    images_dir = yolo_dataset_dir / "images"
    labels_dir = yolo_dataset_dir / "labels"
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    if len(image_files) == 0:
        raise ValueError(f"Не найдено изображений в {images_dir}")
    
    total = len(image_files)
    train_count = int(total * train_split)
    val_count = int(total * val_split)
    test_count = total - train_count - val_count
    
    print(f"Всего изображений: {total}")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    dataset_dir = yolo_dataset_dir.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_images_dir = dataset_dir / split_name / "images"
        split_labels_dir = dataset_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            shutil.copy2(img_file, split_images_dir / img_file.name)
            
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, split_labels_dir / label_file.name)
    
    classes_file = yolo_dataset_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        mapping_file = yolo_dataset_dir / "class_mapping.txt"
        if mapping_file.exists():
            import pandas as pd
            df = pd.read_csv(mapping_file)
            classes = sorted(df['class_name'].unique().tolist())
        else:
            raise FileNotFoundError("Не найден файл classes.txt или class_mapping.txt")
    
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nДатасет подготовлен: {dataset_dir}")
    print(f"Конфигурация: {yaml_path}")
    print(f"Классов: {len(classes)}")
    
    return dataset_dir, yaml_path

def train_model(
    data_yaml_path,
    model_size='n',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',
    project_dir='runs/detect',
    pretrained_path: str | None = None
):
    """Обучает модель YOLOv8 с transfer learning"""
    print(f"\n{'='*60}")
    print("Начало обучения")
    print(f"{'='*60}")
    
    if pretrained_path:
        print(f"Загрузка весов: {pretrained_path}")
        model = YOLO(pretrained_path)
        model_name = Path(pretrained_path).name
    else:
        model_name = f'yolov8{model_size}.pt'
        print(f"Загрузка модели: {model_name}")
        model = YOLO(model_name)
    
    print(f"\nПараметры:")
    print(f"  Модель: {model_name}")
    print(f"  Эпохи: {epochs}")
    print(f"  Батч: {batch}")
    print(f"  Устройство: {device}")
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name='sign_language_detection',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print("Обучение завершено")
    print(f"{'='*60}")
    print(f"Модель сохранена: {results.save_dir}")
    
    return model, results

if __name__ == "__main__":
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    yolo_dataset_dir = script_dir / "yolo_dataset"
    dataset_dir = script_dir / "dataset"
    data_yaml_path = dataset_dir / "data.yaml"
    
    if not data_yaml_path.exists():
        print("Подготовка датасета...")
        dataset_dir, data_yaml_path = prepare_dataset(yolo_dataset_dir)
    else:
        print(f"Используется датасет: {dataset_dir}")
    
    model_size = 'n'
    epochs = 100
    batch = 8
    
    # Автоматический выбор устройства
    import torch
    if torch.cuda.is_available():
        device = '0'
        print(f"\nИспользуется GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("\nИспользуется CPU")
    
    # Поиск существующих весов для продолжения обучения
    pretrained_path = None
    detect_root = script_dir / "runs" / "detect"
    if detect_root.exists():
        for p in sorted(detect_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and p.name.startswith("sign_language_detection"):
                weights_dir = p / "weights"
                if (weights_dir / "best.pt").exists():
                    pretrained_path = str(weights_dir / "best.pt")
                    break
                elif (weights_dir / "last.pt").exists():
                    pretrained_path = str(weights_dir / "last.pt")
                    break
    
    model, results = train_model(
        data_yaml_path=data_yaml_path,
        model_size=model_size,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        project_dir="runs/detect",
        pretrained_path=pretrained_path
    )
    
    print(f"\nМодель сохранена: {results.save_dir / 'weights' / 'best.pt'}")

