"""
Скрипт для тестирования обученной модели детектирования объектов
"""
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

def load_class_names(classes_file):
    """Загружает имена классов из файла"""
    if not Path(classes_file).exists():
        # Пытаемся найти в других местах
        script_dir = Path(__file__).parent
        classes_file = script_dir / "yolo_dataset" / "classes.txt"
        if not classes_file.exists():
            classes_file = script_dir / "dataset" / "classes.txt"
    
    if Path(classes_file).exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    return None

def test_model(
    model_path,
    test_images_dir=None,
    conf_threshold=0.45,
    iou_threshold=0.45,
    save_results=True,
    output_dir="test_results"
):
    """
    Тестирует модель на изображениях
    
    Args:
        model_path: путь к обученной модели (.pt файл)
        test_images_dir: путь к папке с тестовыми изображениями
        conf_threshold: порог уверенности для детекций
        iou_threshold: порог IoU для NMS
        save_results: сохранять ли результаты
        output_dir: папка для сохранения результатов
    """
    print(f"\n{'='*60}")
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print(f"{'='*60}")
    
    # Загружаем модель
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    # Загружаем имена классов
    script_dir = Path(__file__).parent
    classes = load_class_names(script_dir / "yolo_dataset" / "classes.txt")
    if classes:
        print(f"Загружено классов: {len(classes)}")
        print(f"Классы: {classes[:10]}..." if len(classes) > 10 else f"Классы: {classes}")
    
    # Определяем папку с тестовыми изображениями
    if test_images_dir is None:
        # Пытаемся найти тестовую папку
        dataset_dir = script_dir / "dataset" / "test" / "images"
        archive_dir = script_dir / "archive" / "images"
        
        if dataset_dir.exists():
            test_images_dir = dataset_dir
        elif archive_dir.exists():
            test_images_dir = archive_dir
        else:
            raise ValueError("Не найдена папка с тестовыми изображениями")
    
    test_images_dir = Path(test_images_dir)
    print(f"Тестовая папка: {test_images_dir}")
    
    # Получаем список изображений
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        raise ValueError(f"Не найдено изображений в {test_images_dir}")
    
    print(f"Найдено изображений для тестирования: {len(image_files)}")
    
    # Ограничиваем количество для демонстрации (можно убрать)
    max_images = min(20, len(image_files))
    image_files = image_files[:max_images]
    print(f"Будет обработано изображений: {len(image_files)}")
    
    # Создаем папку для результатов
    if save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"Результаты будут сохранены в: {output_dir}")
    
    # Обрабатываем изображения
    all_results = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\nОбработка {i}/{len(image_files)}: {img_path.name}")
        
        # Предсказание
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        result = results[0]
        
        # Получаем информацию о детекциях
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # координаты в формате (x1, y1, x2, y2)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            print(f"  Найдено объектов: {len(boxes)}")
            
            for j, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = classes[cls_id] if classes and cls_id < len(classes) else f"class_{cls_id}"
                print(f"    {j+1}. {class_name}: {conf:.2f}")
                
                all_results.append({
                    'image': img_path.name,
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                })
        else:
            print(f"  Объекты не обнаружены")
        
        # Сохраняем визуализацию
        if save_results:
            annotated_img = result.plot()  # YOLO автоматически рисует боксы
            output_path = output_dir / f"result_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
    
    # Сохраняем сводку результатов
    if save_results and all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = output_dir / "detections.csv"
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nРезультаты сохранены в: {csv_path}")
        
        # Статистика
        print(f"\n{'='*60}")
        print("СТАТИСТИКА ДЕТЕКЦИЙ")
        print(f"{'='*60}")
        print(f"Всего детекций: {len(all_results)}")
        if classes:
            print(f"\nДетекции по классам:")
            class_counts = df_results['class_name'].value_counts()
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}")
    
    print(f"\n✅ Тестирование завершено!")
    return all_results

def evaluate_model(model_path, data_yaml_path):
    """
    Оценивает модель на валидационном/тестовом наборе
    
    Args:
        model_path: путь к обученной модели
        data_yaml_path: путь к data.yaml с конфигурацией датасета
    """
    print(f"\n{'='*60}")
    print("ОЦЕНКА МОДЕЛИ (МЕТРИКИ)")
    print(f"{'='*60}")
    
    model = YOLO(model_path)
    
    # Валидация модели
    metrics = model.val(data=str(data_yaml_path))
    
    print(f"\nРезультаты валидации:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    return metrics

if __name__ == "__main__":
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    
    # Путь к обученной модели
    # Ищем лучшую модель в последних запусках (sign_language_detection, sign_language_detection2, ...)
    detect_root = script_dir / "runs" / "detect"
    candidates = []
    if detect_root.exists():
        # Берём все подпапки вида sign_language_detection*
        for p in sorted(detect_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and p.name.startswith("sign_language_detection"):
                weights_dir = p / "weights"
                if (weights_dir / "best.pt").exists():
                    candidates.append(weights_dir / "best.pt")
                elif (weights_dir / "last.pt").exists():
                    candidates.append(weights_dir / "last.pt")
    model_path = candidates[0] if candidates else None
    
    if model_path is None or not model_path.exists():
        print("Ошибка: не найдена обученная модель!")
        print("Сначала запустите train_yolo.py для обучения модели")
        print(f"Искали в подпапках: {detect_root}/sign_language_detection*")
        exit(1)
    
    print(f"Используется модель: {model_path}")
    
    # Тестируем модель на нескольких изображениях
    test_model(
        model_path=str(model_path),
        conf_threshold=0.25,
        iou_threshold=0.45,
        save_results=True,
        output_dir="test_results"
    )
    
    # Оцениваем модель на тестовом наборе (если есть)
    data_yaml_path = script_dir / "dataset" / "data.yaml"
    if data_yaml_path.exists():
        print("\n" + "="*60)
        evaluate_model(str(model_path), data_yaml_path)

