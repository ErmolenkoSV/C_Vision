import cv2
import numpy as np
import os
from pathlib import Path

class TomatoClassifier:
    def __init__(self):

        # Пороги для классификации (будут вычислены автоматически)
        self.thresholds = {
            'fully_ripened': 0.50,  # По умолчанию 50%
            'half_ripened': 0.05,   # По умолчанию 5%
            'green': 0.0
        }
        
        # соответствие категорий на номера
        self.category_mapping = {
            'fully_ripened': 1,
            'half_ripened': 2,
            'green': 3
        }
        
        # Флаг: пороги вычислены автоматически?
        self.thresholds_calculated = False
    
    def calculate_red_percentage(self, image_path):
        
        
        # ШАГ 1: Загружаем изображение
        # -----------------------------
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Ошибка загрузки: {image_path}")
            return 0.0
        
        # ШАГ 2: Преобразуем BGR → HSV
        # Преобразуем в HSV для удобной работы с цветом
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # ШАГ 3: Создаем маску для КРАСНОГО цвета
        # Hue: 0-180 (цвет)
        # Saturation: 0-255 (насыщенность, 0=серый, 255=яркий)
        # Value: 0-255 (яркость, 0=черный, 255=белый)
        
        # ДИАПАЗОН 1: Темно-красный (H = 0-10°)
        lower_red1 = np.array([0, 50, 50])    # [Hue=0, Sat>=50, Val>=50]
        upper_red1 = np.array([10, 255, 255]) # [Hue=10, Sat<=255, Val<=255]
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # ДИАПАЗОН 2: Ярко-красный (H = 170-180°)
        lower_red2 = np.array([170, 50, 50])   
        upper_red2 = np.array([180, 255, 255]) 
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        # mask2 = бинарная маска (255=красный, 0=не красный)
        
        red_mask = mask1 + mask2
      
        
        # ШАГ 4:  процент красных пикселей
      
        total_pixels = img.shape[0] * img.shape[1] 
        red_pixels = np.sum(red_mask > 0)           
        red_percentage = red_pixels / total_pixels 
        
       
        return red_percentage
    
    def classify_tomato(self, image_path):
       
        # Вычисляем процент красного цвета
        red_percentage = self.calculate_red_percentage(image_path)
        
        # Определяем категорию на основе процента (проверяем от большего к меньшему!)
        if red_percentage >= self.thresholds['fully_ripened']:
            # >= 50% красного → полностью созревший
            category = 'fully_ripened'
        elif red_percentage >= self.thresholds['half_ripened']:
            # >= 5%, но < 50% красного → полусозревший
            category = 'half_ripened'
        else:
            # < 5% красного → зеленый
            category = 'green'
        
        # Преобразуем название категории в номер (1, 2 или 3)
        category_number = self.category_mapping[category]
        
        return category_number, red_percentage
    
    def calculate_optimal_thresholds_method1(self, stats):
        
        # МЕТОД 1: Среднее между границами категорий
        #Порог = (max(предыдущей) + min(следующей)) / 2
        
        thresholds = {}
        
        if 'green' in stats and 'half_ripened' in stats:
            thresholds['half_ripened'] = (stats['green']['max'] + stats['half_ripened']['min']) / 2
        
        if 'half_ripened' in stats and 'fully_ripened' in stats:
            thresholds['fully_ripened'] = (stats['half_ripened']['max'] + stats['fully_ripened']['min']) / 2
        
        return thresholds
    
    def calculate_optimal_thresholds_method2(self, stats):
        
        # МЕТОД 2: Минимизация ошибок классификации
        #Перебираем пороги и ищем тот, при котором меньше всего ошибок
        
        thresholds = {}
        
        # Для half_ripened -> fully_ripened
        if 'half_ripened' in stats and 'fully_ripened' in stats:
            best_threshold = None
            min_errors = float('inf')
            
            # Перебираем пороги с шагом 1%
            for threshold in range(
                int(stats['half_ripened']['min'] * 100), 
                int(stats['fully_ripened']['max'] * 100), 
                1
            ):
                threshold_value = threshold / 100
                errors = 0
                
                # Сколько half_ripened попадет в fully_ripened?
                for val in stats['half_ripened']['values']:
                    if val >= threshold_value:
                        errors += 1
                
                # Сколько fully_ripened попадет в half_ripened?
                for val in stats['fully_ripened']['values']:
                    if val < threshold_value:
                        errors += 1
                
                if errors < min_errors:
                    min_errors = errors
                    best_threshold = threshold_value
            
            thresholds['fully_ripened'] = best_threshold
            thresholds['errors'] = min_errors
        
        # Для green -> half_ripened используем метод 1
        if 'green' in stats and 'half_ripened' in stats:
            thresholds['half_ripened'] = (stats['green']['max'] + stats['half_ripened']['min']) / 2
        
        return thresholds
    
    def analyze_training_data(self, train_dir):
        
        #Анализ обучающих данных для калибровки порогов
        
        print("\nАНАЛИЗ ОБУЧАЮЩИХ ДАННЫХ:")
        
        categories = ['fully_ripened', 'half_ripened', 'green']
        statistics = {}
        
        for category in categories:
            category_path = Path(train_dir) / category
            if not category_path.exists():
                print(f"ВНИМАНИЕ: Папка {category} не найдена")
                continue
            
            images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.png'))
            red_percentages = []
            
            for img_path in images:
                red_pct = self.calculate_red_percentage(img_path)
                red_percentages.append(red_pct)
            
            if red_percentages:
                statistics[category] = {
                    'count': len(red_percentages),
                    'mean': np.mean(red_percentages),
                    'min': np.min(red_percentages),
                    'max': np.max(red_percentages),
                    'std': np.std(red_percentages),
                    'values': red_percentages  # Сохраняем значения для метода 2
                }
                
                print(f"\n{category}:")
                print(f"  Количество изображений: {statistics[category]['count']}")
                print(f"  Средний % красного: {statistics[category]['mean']*100:.1f}%")
                print(f"  Диапазон: {statistics[category]['min']*100:.1f}% - {statistics[category]['max']*100:.1f}%")
                print(f"  Стандартное отклонение: {statistics[category]['std']*100:.1f}%")
        
        # Вычисляем пороги двумя методами
        if statistics and len(statistics) == 3:
            print("\n" + "="*60)
            print("ВЫЧИСЛЕНИЕ ОПТИМАЛЬНЫХ ПОРОГОВ")
            print("="*60)
            
            # МЕТОД 1: Среднее между границами
            thresholds_m1 = self.calculate_optimal_thresholds_method1(statistics)
            print("\nМЕТОД 1: Среднее между границами категорий")
            if 'half_ripened' in thresholds_m1:
                print(f"  green -> half_ripened: {thresholds_m1['half_ripened']*100:.1f}%")
            if 'fully_ripened' in thresholds_m1:
                print(f"  half_ripened -> fully_ripened: {thresholds_m1['fully_ripened']*100:.1f}%")
                print(f"    (среднее между {statistics['half_ripened']['max']*100:.1f}% и {statistics['fully_ripened']['min']*100:.1f}%)")
            
            # МЕТОД 2: Минимизация ошибок
            thresholds_m2 = self.calculate_optimal_thresholds_method2(statistics)
            print("\nМЕТОД 2: Минимизация ошибок классификации")
            if 'half_ripened' in thresholds_m2:
                print(f"  green -> half_ripened: {thresholds_m2['half_ripened']*100:.1f}%")
            if 'fully_ripened' in thresholds_m2:
                print(f"  half_ripened -> fully_ripened: {thresholds_m2['fully_ripened']*100:.1f}%")
                print(f"    (ошибок на обучающих данных: {thresholds_m2.get('errors', 'N/A')})")
            
            # Выбираем финальные пороги (используем метод 2 как более точный)
            print("\n" + "="*60)
            print("ВЫБРАННЫЕ ПОРОГИ")
            print("="*60)
            
            if 'fully_ripened' in thresholds_m2:
                self.thresholds['fully_ripened'] = thresholds_m2['fully_ripened']
                self.thresholds['half_ripened'] = thresholds_m2.get('half_ripened', 0.05)
                self.thresholds_calculated = True
                
                print(f"\nИспользуем МЕТОД 2 (минимизация ошибок):")
                print(f"  green -> half_ripened: {self.thresholds['half_ripened']*100:.1f}%")
                print(f"  half_ripened -> fully_ripened: {self.thresholds['fully_ripened']*100:.1f}%")
                print(f"\nОбоснование:")
                print(f"  - Метод 2 дает {thresholds_m2.get('errors', 0)} ошибок на обучающих данных")
                print(f"  - Это оптимальный порог для разделения категорий")
            else:
                print("\nИспользуем пороги по умолчанию (50% и 5%)")
                print("  (недостаточно данных для автоматического вычисления)")
        
        return statistics
    
    def classify_test_images(self, test_dir, output_file='answer.txt'):
        """
        Классификация всех тестовых изображений
        """
        print("\nКЛАССИФИКАЦИЯ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ:")
        print("="*60)
        
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"ОШИБКА: Папка {test_dir} не найдена!")
            return
        
        results = []
        
        # Обрабатываем изображения от 001.jpg до 100.jpg
        for i in range(1, 101):
            img_name = f"{i:03d}.jpg"
            img_path = test_path / img_name
            
            if img_path.exists():
                category_num, red_pct = self.classify_tomato(img_path)
                results.append(category_num)
                
                # Выводим прогресс каждые 10 изображений
                if i % 10 == 0:
                    print(f"Обработано: {i}/100 изображений")
            else:
                print(f"ВНИМАНИЕ: Файл {img_name} не найден, используем категорию по умолчанию")
                results.append(2)  # По умолчанию - half_ripened
        
        # Сохраняем результаты в файл
        output_path = Path(test_dir).parent / output_file
        with open(output_path, 'w') as f:
            for category in results:
                f.write(f"{category}\n")
        
        print(f"\nГОТОВО! Результаты сохранены в {output_path}")
        print(f"Всего классифицировано: {len(results)} изображений")
        
        # Статистика по категориям
        from collections import Counter
        category_counts = Counter(results)
        print("\nРАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
        print(f"  Категория 1 (fully_ripened): {category_counts[1]} томатов")
        print(f"  Категория 2 (half_ripened): {category_counts[2]} томатов")
        print(f"  Категория 3 (green): {category_counts[3]} томатов")
        
        return results

def main():
    print("КЛАССИФИКАЦИИ ТОМАТОВ ПО СТЕПЕНИ ЗРЕЛОСТИ")
    print("="*60)
    
    classifier = TomatoClassifier()
    
    # Пути к данным
    # Определяем директорию, где находится скрипт task1.py
    script_dir = Path(__file__).parent  # VISION_C/
    
    # Ищем папки train и test в подпапке task1/
    base_dir = script_dir / "task1"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    output_file = "answer.txt"
        
    # Шаг 1: Анализ обучающих данных 
    if train_dir.exists():
        print("\nОбнаружена папка train/ - анализирую обучающие данные...")
        statistics = classifier.analyze_training_data(train_dir)
    else:
        print(f"\nПапка train/ не найдена")
        print("Используются стандартные пороги: 30% и 90% красного цвета")
    
    # Шаг 2: Классификация тестовых изображений
    if test_dir.exists():
        results = classifier.classify_test_images(test_dir, output_file)
    else:
        print(f"\nОШИБКА: Папка {test_dir} не найдена!")
        print("Создайте папку 'test' с изображениями томатов.")
        return
    

if __name__ == "__main__":
    main()
