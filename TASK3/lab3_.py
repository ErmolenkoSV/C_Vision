"""
Лабораторная работа 3: Построение панорамных изображений
Реализация алгоритма склейки панорам без использования готового класса cv2.Stitcher

Алгоритм:
1. Поиск ключевых точек и дескрипторов (SIFT/ORB)
2. Сопоставление признаков с фильтрацией (ratio-test)
3. Оценка гомографии с помощью RANSAC
4. Проективное преобразование и склейка изображений
5. Градиентный blending в зоне перекрытия
"""

import cv2
import numpy as np
import os

# Путь к папке с исходными изображениями
# Используем путь относительно расположения скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "task3")

def detect_and_describe(image, num_features=5000):
    """
    Находит особые точки (keypoints) и их дескрипторы на изображении.
    Использует SIFT (масштабно-инвариантные признаки) - это изученный материал.
    Если SIFT недоступен, использует ORB как запасной вариант.
    """
    # Переводим в оттенки серого для работы детекторов
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Улучшаем контраст - это помогает найти больше точек на однородных участках
    gray = cv2.equalizeHist(gray)

    # Пробуем использовать SIFT (вопрос 8 из экзамена)
    try:
        sift = cv2.SIFT_create(nfeatures=num_features, contrastThreshold=0.02, edgeThreshold=20)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None and len(keypoints) > 0:
            return keypoints, descriptors
    except Exception:
        pass

    # Если SIFT не работает, используем ORB
    orb = cv2.ORB_create(nfeatures=num_features, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio=0.65):
    """
    Сопоставляет дескрипторы двух изображений.
    Использует KNN-матчинг и ratio-test (Lowe's ratio test) для фильтрации ложных совпадений.
    Это стандартный метод для отсеивания плохих матчей.
    """
    # Выбираем метрику расстояния в зависимости от типа дескрипторов
    if desc1.dtype == np.float32:  # SIFT использует L2-норму
        norm_type = cv2.NORM_L2
    else:  # ORB использует Hamming расстояние
        norm_type = cv2.NORM_HAMMING
    
    # Brute-Force матчер для поиска соответствий
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    # Фильтруем матчи по ratio-test: оставляем только те, где лучший матч
    # значительно лучше второго (меньше ratio - строже фильтрация)
    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair  # m - лучший, n - второй по качеству
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    # Сортируем по качеству (лучшие матчи первыми)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches

def get_matched_points(kp1, kp2, matches, max_matches=200):
    """
    Берёт первые max_matches лучших совпадений и строит из них
    массивы координат точек на первом и втором изображениях.
    Увеличено до 200 для максимальной точности оценки гомографии.
    Формат вывода: pts1 и pts2 формы (N, 1, 2), dtype=float32.
    """
    matches = matches[:min(max_matches, len(matches))]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2

def estimate_homography(pts1, pts2):
    """
    Оценивает гомографию (матрицу проективного преобразования 3x3) по парам точек.
    Использует RANSAC для отсеивания выбросов (outliers) - это важно для устойчивости.
    Гомография показывает, как нужно "перегнуть" второе изображение, чтобы оно совпало с первым.
    """
    # RANSAC с порогом 3 пикселя - компромисс между точностью и устойчивостью
    H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
    
    # Если нашли достаточно надёжных точек (inliers), переоцениваем гомографию только на них
    # Это даёт более точную матрицу
    if H is not None and status is not None:
        inlier_count = np.sum(status)
        if inlier_count >= 8:  # минимум 8 точек для гомографии
            inlier_pts1 = pts1[status.ravel() == 1]
            inlier_pts2 = pts2[status.ravel() == 1]
            # Более строгий порог для финальной оценки
            H_refined, _ = cv2.findHomography(inlier_pts2, inlier_pts1, cv2.RANSAC, 2.0)
            if H_refined is not None:
                return H_refined, status
    
    return H, status

def estimate_affine_fallback(pts1, pts2):
    """
    Запасной вариант: если гомография не работает (мало точек или плохое качество),
    пробуем более простое аффинное преобразование.
    Аффинное преобразование проще гомографии (нет перспективы), но более устойчиво.
    """
    M, inliers = cv2.estimateAffinePartial2D(
        pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return None, None
    # Преобразуем аффинную матрицу 2x3 в гомографию 3x3
    H_affine = np.vstack([M, [0, 0, 1]])
    return H_affine, inliers

def is_good_homography(H, img1_shape, img2_shape):
    """
    Проверяет, что гомография не делает изображение слишком маленьким или большим.
    Это защита от "диких" преобразований, которые могут испортить панораму.
    """
    if H is None:
        return False
    
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    
    # Углы второго изображения
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)
    
    # Проверяем, что углы не слишком далеко от первого изображения
    # (это значит, что изображения действительно перекрываются)
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Проверяем, что хотя бы один угол второго изображения попадает в разумные границы
    x_coords = corners2_transformed[:, 0, 0]
    y_coords = corners2_transformed[:, 0, 1]
    
    # Проверяем, что преобразование не делает изображение слишком маленьким или большим
    width_transformed = np.max(x_coords) - np.min(x_coords)
    height_transformed = np.max(y_coords) - np.min(y_coords)
    
    # Если преобразованное изображение слишком большое или слишком маленькое - плохо
    if width_transformed > w1 * 5 or height_transformed > h1 * 5:
        return False
    if width_transformed < w1 * 0.1 or height_transformed < h1 * 0.1:
        return False
    
    return True


def crop_black_borders(image, threshold=15):
    """
    Обрезает чёрные поля по краям панорамы.
    Это нужно, потому что после проективного преобразования появляются пустые области.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # маска всех не-чёрных пикселей
    mask = gray > threshold
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # +1, чтобы включить последнюю строку/столбец
    return image[y_min:y_max, x_min:x_max]

def assess_pair_quality(img1, img2):
    """
    Оценивает, насколько хорошо два изображения перекрываются.
    Нужно для выбора оптимального порядка склейки - начинаем с лучших пар.
    """
    kp1, desc1 = detect_and_describe(img1, num_features=1000)
    kp2, desc2 = detect_and_describe(img2, num_features=1000)
    
    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0, 0.0
    
    matches = match_features(desc1, desc2, ratio=0.7)
    
    if len(matches) < 4:
        return 0, 0.0
    
    # Средняя дистанция матчей (меньше - лучше)
    avg_distance = np.mean([m.distance for m in matches[:50]])
    quality_score = len(matches) / (1.0 + avg_distance)
    
    return len(matches), quality_score

def stitch_two_images(img1, img2):
    """
    Основная функция склейки двух изображений.
    Алгоритм:
    1. Находим особые точки на обоих изображениях
    2. Сопоставляем их (matching)
    3. Оцениваем гомографию по найденным соответствиям
    4. Применяем проективное преобразование (warpPerspective)
    5. Смешиваем изображения в зоне перекрытия (blending)
    """
    # Шаг 1: Находим особые точки и их описания (дескрипторы)
    kp1, desc1 = detect_and_describe(img1)
    kp2, desc2 = detect_and_describe(img2)

    if desc1 is None or desc2 is None:
        print("Не удалось найти дескрипторы на одном из изображений.")
        return img1

    # Шаг 2: Сопоставляем дескрипторы (ищем одинаковые точки на двух изображениях)
    matches = match_features(desc1, desc2, ratio=0.7)
    if len(matches) < 15:
        print(f"Слишком мало совпадений: {len(matches)}")
        return img1

    # Шаг 3: Извлекаем координаты найденных пар точек
    pts1, pts2 = get_matched_points(kp1, kp2, matches, max_matches=100)

    # Шаг 4: Оцениваем гомографию (матрицу преобразования)
    H, status = estimate_homography(pts1, pts2)

    # Проверяем качество гомографии
    inlier_count = 0
    if status is not None:
        inlier_count = int(np.sum(status))

    # Если гомография плохая, пробуем более простое аффинное преобразование
    if H is None or inlier_count < 8:
        print(f"Гомография слабая (inliers={inlier_count}), пробуем аффинный fallback...")
        H_affine, aff_inliers = estimate_affine_fallback(pts1, pts2)
        if H_affine is not None:
            H = H_affine
            status = aff_inliers
            inlier_count = int(np.sum(aff_inliers)) if aff_inliers is not None else inlier_count
        else:
            print("Не удалось оценить преобразование.")
            return img1

    # Шаг 5: Вычисляем размеры холста для итоговой панорамы
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Углы второго изображения
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    # Углы первого изображения
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)

    # Преобразуем углы второго изображения в систему координат первого
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # Собираем все углы и ищем минимальные/максимальные координаты
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Размер будущего холста
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    # Матрица сдвига, чтобы всё поместилось в положительные координаты
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    # Шаг 6: Применяем проективное преобразование ко второму изображению
    warped_img2 = cv2.warpPerspective(img2, translation @ H, (canvas_width, canvas_height))

    # Шаг 7: Создаём итоговое изображение
    result = warped_img2.copy()
    
    # Вычисляем смещения для размещения первого изображения
    x_offset = -x_min
    y_offset = -y_min
    
    # Шаг 8: Определяем зону перекрытия для плавного смешивания
    warped_region = warped_img2[y_offset:y_offset + h1, x_offset:x_offset + w1]
    mask_warped = (warped_region.sum(axis=2) > 10)
    mask_img1 = np.ones((h1, w1), dtype=bool)
    overlap = mask_warped & mask_img1
    
    # Шаг 9: Плавное смешивание (blending) в зоне перекрытия
    # Используем градиентный переход, чтобы швы были незаметными
    if np.any(overlap):
        img1_pixels = img1.astype(np.float32)
        warped_pixels = warped_region.astype(np.float32)
        
        # Создаём градиентную маску: слева больше вес первого изображения,
        # справа - второго. Это делает переход плавным.
        alpha_mask = np.zeros((h1, w1), dtype=np.float32)
        
        # Находим границы зоны перекрытия
        overlap_cols = np.where(np.any(overlap, axis=0))[0]
        if len(overlap_cols) > 0:
            left_col = overlap_cols[0]
            right_col = overlap_cols[-1]
            overlap_width = right_col - left_col + 1
            
            # Создаём градиент от 1.0 (слева) до 0.0 (справа)
            for col in range(left_col, right_col + 1):
                weight = 1.0 - (col - left_col) / max(overlap_width, 1)
                alpha_mask[:, col] = weight
        
        # Применяем взвешенное усреднение для каждого цветового канала
        blended = img1_pixels.copy()
        for c in range(3):  # B, G, R каналы
            blended[:, :, c][overlap] = (
                img1_pixels[:, :, c][overlap] * alpha_mask[overlap] +
                warped_pixels[:, :, c][overlap] * (1.0 - alpha_mask[overlap])
            )
        
        result[y_offset:y_offset + h1, x_offset:x_offset + w1] = blended.astype(np.uint8)
    else:
        # Если перекрытия нет, просто кладём первое изображение поверх
        result[y_offset:y_offset + h1, x_offset:x_offset + w1] = img1

    return result

def build_panorama_for_folder(folder_name, output_name):
    """
    Строит панораму для набора изображений в папке.
    Использует умную стратегию: сначала оценивает качество всех пар,
    затем выбирает лучший стартовый кадр и склеивает в обе стороны от него.
    Это минимизирует накопление ошибок.
    """
    folder_path = os.path.join(BASE_DIR, folder_name)
    files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    if not files:
        print(f"В папке {folder_path} нет изображений.")
        return

    # Читаем изображения
    images = []
    filenames = []
    for fname in files:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось прочитать {img_path}")
            continue
        images.append(img)
        filenames.append(fname)

    if len(images) < 2:
        print(f"Недостаточно изображений в {folder_path}")
        return

    print(f"\n=== Обработка {folder_name}: {len(images)} изображений ===")
    
    # Оцениваем качество перекрытия всех соседних пар
    # Это нужно, чтобы понять, с какого кадра лучше начинать склейку
    print("Оценка качества пар...")
    pair_qualities = []
    for i in range(len(images) - 1):
        num_matches, quality = assess_pair_quality(images[i], images[i+1])
        pair_qualities.append((i, i+1, num_matches, quality))
        print(f"  {filenames[i]} <-> {filenames[i+1]}: {num_matches} матчей, quality={quality:.2f}")
    
    # Находим лучший стартовый кадр - тот, у которого суммарное качество связей максимально
    # Это кадр, который лучше всего "стыкуется" с соседями
    node_scores = [0.0] * len(images)
    for i, j, matches, quality in pair_qualities:
        node_scores[i] += quality
        node_scores[j] += quality
    
    best_start_idx = np.argmax(node_scores)
    print(f"\nЛучший стартовый кадр: {filenames[best_start_idx]} (score={node_scores[best_start_idx]:.2f})")
    
    # Начинаем склейку с лучшего кадра и идём в обе стороны
    # Это уменьшает накопление ошибок по сравнению с простой последовательной склейкой
    panorama = images[best_start_idx]
    used = {best_start_idx}
    
    # Склеиваем кадры справа от стартового
    current_idx = best_start_idx
    while current_idx + 1 < len(images):
        next_idx = current_idx + 1
        if next_idx not in used:
            print(f"Склейка: добавляем {filenames[next_idx]} справа...")
            panorama = stitch_two_images(panorama, images[next_idx])
            used.add(next_idx)
        current_idx += 1
    
    # Склеиваем кадры слева от стартового
    current_idx = best_start_idx
    while current_idx - 1 >= 0:
        prev_idx = current_idx - 1
        if prev_idx not in used:
            print(f"Склейка: добавляем {filenames[prev_idx]} слева...")
            # Для склейки слева меняем порядок аргументов
            panorama = stitch_two_images(images[prev_idx], panorama)
            used.add(prev_idx)
        current_idx -= 1
    
    # Обрезаем чёрные поля по краям
    panorama_cropped = crop_black_borders(panorama, threshold=15)
    
    # Сохраняем
    out_path = os.path.join(BASE_DIR, output_name)
    cv2.imwrite(out_path, panorama_cropped)
    print(f"\n✓ Панорама для {folder_name} сохранена в {out_path}\n")

if __name__ == "__main__":
    """
    Основная программа: строит панорамы для трёх наборов изображений.
    Результаты сохраняются в папке task3 как panorama_map1.jpg, panorama_map2.jpg, panorama_map3.jpg
    """
    print("=" * 60)
    print("Лабораторная работа 3: Построение панорамных изображений")
    print("=" * 60)
    
    build_panorama_for_folder("map1", "panorama_map1.jpg")
    build_panorama_for_folder("map2", "panorama_map2.jpg")
    build_panorama_for_folder("map3", "panorama_map3.jpg")
    
    print("=" * 60)
    print("Готово! Все панорамы сохранены в папке task3/")
    print("=" * 60)

