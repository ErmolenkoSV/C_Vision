"""
Скрипт для обучения модели детектирования объектов с использованием Transfer Learning
Использует YOLOv8 с предобученными весами
"""
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil

def prepare_dataset(yolo_dataset_dir, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Подготавливает датасет для YOLO: разделяет на train/val/test
    
    Args:
        yolo_dataset_dir: путь к папке с YOLO датасетом
        train_split: доля обучающей выборки
        val_split: доля валидационной выборки
        test_split: доля тестовой выборки
    """
    yolo_dataset_dir = Path(yolo_dataset_dir)
    
    # Проверяем, что сумма равна 1
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Сумма долей должна быть равна 1"
    
    images_dir = yolo_dataset_dir / "images"
    labels_dir = yolo_dataset_dir / "labels"
    
    # Получаем список всех изображений
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    if len(image_files) == 0:
        raise ValueError(f"Не найдено изображений в {images_dir}")
    
    total = len(image_files)
    train_count = int(total * train_split)
    val_count = int(total * val_split)
    test_count = total - train_count - val_count
    
    print(f"Всего изображений: {total}")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    
    # Разделяем файлы
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Создаем структуру папок
    dataset_dir = yolo_dataset_dir.parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_images_dir = dataset_dir / split_name / "images"
        split_labels_dir = dataset_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            # Копируем изображение
            shutil.copy2(img_file, split_images_dir / img_file.name)
            
            # Копируем соответствующий файл аннотации
            label_file = labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, split_labels_dir / label_file.name)
    
    # Создаем файл конфигурации для YOLO
    classes_file = yolo_dataset_dir / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    else:
        # Если файла нет, пытаемся определить из class_mapping.txt
        mapping_file = yolo_dataset_dir / "class_mapping.txt"
        if mapping_file.exists():
            import pandas as pd
            df = pd.read_csv(mapping_file)
            classes = sorted(df['class_name'].unique().tolist())
        else:
            raise FileNotFoundError("Не найден файл classes.txt или class_mapping.txt")
    
    # Создаем data.yaml
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
    
    print(f"\nДатасет подготовлен в: {dataset_dir}")
    print(f"Конфигурация сохранена в: {yaml_path}")
    print(f"Количество классов: {len(classes)}")
    
    return dataset_dir, yaml_path

def train_model(
    data_yaml_path,
    model_size='n',  # n=nano, s=small, m=medium, l=large, x=xlarge
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',  # 'cpu' или '0' для GPU
    project_dir='runs/detect',
    pretrained_path: str | None = None  # путь к своим весам для дообучения
):
    """
    Обучает модель YOLOv8 с transfer learning
    
    Args:
        data_yaml_path: путь к файлу data.yaml с конфигурацией датасета
        model_size: размер модели ('n', 's', 'm', 'l', 'x')
        epochs: количество эпох обучения
        imgsz: размер изображения для обучения
        batch: размер батча
        device: устройство для обучения ('cpu' или '0', '1', etc. для GPU)
        project_dir: директория для сохранения результатов
    """
    print(f"\n{'='*60}")
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    print(f"{'='*60}")
    
    # Загружаем модель: либо свои веса (если переданы), либо предобученную YOLO
    if pretrained_path:
        print(f"Загрузка своих весов для дообучения: {pretrained_path}")
        model = YOLO(pretrained_path)
        model_name = Path(pretrained_path).name
    else:
        model_name = f'yolov8{model_size}.pt'
        print(f"Загрузка предобученной модели: {model_name}")
        model = YOLO(model_name)
    
    # Обучаем модель
    print(f"\nПараметры обучения:")
    print(f"  - Модель: {model_name}")
    print(f"  - Эпохи: {epochs}")
    print(f"  - Размер изображения: {imgsz}")
    print(f"  - Размер батча: {batch}")
    print(f"  - Устройство: {device}")
    print(f"  - Датасет: {data_yaml_path}")
    
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name='sign_language_detection',
        save=True,
        save_period=10,  # Сохранять чекпоинт каждые 10 эпох
        val=True,  # Валидация во время обучения
        plots=True,  # Генерировать графики
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Лучшая модель сохранена в: {results.save_dir}")
    
    return model, results

if __name__ == "__main__":
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    yolo_dataset_dir = script_dir / "yolo_dataset"
    
    # Проверяем, существует ли уже подготовленный датасет
    dataset_dir = script_dir / "dataset"
    data_yaml_path = dataset_dir / "data.yaml"
    
    if not data_yaml_path.exists():
        print("Подготовка датасета...")
        dataset_dir, data_yaml_path = prepare_dataset(yolo_dataset_dir)
    else:
        print(f"Используется существующий датасет: {dataset_dir}")
    
    # Параметры обучения
    # Можно изменить размер модели: 'n' (быстро, меньше точность) или 's', 'm' (медленнее, выше точность)
    model_size = 'n'  # nano - быстрая модель для начала
    
    # Для CPU используйте меньший batch и меньше эпох для тестирования
    # Для GPU можно увеличить batch до 32-64 и epochs до 100-200
    epochs = 100  # 100 эпох для хорошего качества (на CPU можно уменьшить до 20-30)
    batch = 8  # Уменьшите для CPU или если не хватает памяти
    
    # Определяем устройство (автоматически выбирает GPU если доступен)
    import torch
    if torch.cuda.is_available():
        device = '0'  # Используем первую GPU
        print(f"\n✅ Найдена GPU: {torch.cuda.get_device_name(0)}")
        print(f"Используемое устройство: GPU (cuda:0)")
    else:
        device = 'cpu'
        print(f"\n⚠️  GPU не найдена. Используется CPU.")
        print("Внимание: обучение на CPU будет медленным (может занять часы).")
        print("Рекомендации:")
        print("  1. Используйте облачный GPU (Google Colab, Kaggle)")
        print("  2. Уменьшите количество эпох (epochs=20-30)")
        print("  3. Уменьшите размер батча (batch=4-8)")
        print("  4. Используйте меньшую модель (model_size='n')")
    
    # Ищем последние веса (best/last) чтобы при наличии дообучать с них
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
    
    # Обучаем модель
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
    
    print(f"\n✅ Обучение завершено успешно!")
    print(f"Модель сохранена в: {results.save_dir / 'weights' / 'best.pt'}")

