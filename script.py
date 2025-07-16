import cv2
import numpy as np
import argparse
import os

def apply_whitening(image: np.ndarray) -> np.ndarray:
    """
    Применяет операцию Whitening к каждому каналу цветного изображения.
    Приводит среднее значение к 0 и стандартное отклонение к 1,
    а затем нормализует результат в диапазон 0-255 для визуализации.

    Args:
        image (np.ndarray): Входное изображение в формате BGR (стандарт для OpenCV).

    Returns:
        np.ndarray: Обработанное изображение в формате BGR.
    """
    # Создаем копию изображения для работы с float
    img_float = image.copy().astype(np.float32)

    # Разделяем на каналы (B, G, R)
    channels = cv2.split(img_float)
    processed_channels = []

    for channel in channels:
        # Рассчитываем среднее и стандартное отклонение
        mean = np.mean(channel)
        std = np.std(channel)

        # Избегаем деления на ноль для каналов с нулевой дисперсией (например, сплошной цвет)
        if std < 1e-6:
            std = 1.0

        # Применяем формулу whitening: (X - mu) / sigma
        whitened_channel = (channel - mean) / std

        # Нормализуем результат обратно в диапазон 0-255 для визуализации
        # cv2.NORM_MINMAX растягивает значения так, что min становится 0, а max - 255.
        cv2.normalize(whitened_channel, whitened_channel, 0, 255, cv2.NORM_MINMAX)

        processed_channels.append(whitened_channel)

    # Собираем каналы обратно в изображение и конвертируем в 8-битный формат
    result_image = cv2.merge(processed_channels)
    return result_image.astype(np.uint8)

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Применяет выравнивание гистограммы к цветному изображению,
    сохраняя при этом цветовой баланс.

    Args:
        image (np.ndarray): Входное изображение в формате BGR.

    Returns:
        np.ndarray: Обработанное изображение в формате BGR.
    """
    # Преобразуем изображение в цветовое пространство YCrCb
    # Y - канал яркости (luma), Cr и Cb - каналы цветности (chroma)
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Разделяем на каналы
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

    # Применяем выравнивание гистограммы ТОЛЬКО к каналу яркости (Y)
    cv2.equalizeHist(y_channel, y_channel)

    # Собираем каналы обратно
    equalized_ycrcb = cv2.merge([y_channel, cr_channel, cb_channel])

    # Конвертируем изображение обратно в BGR
    result_image = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)
    return result_image


def main():
    """
    Главная функция для пакетной обработки изображений.
    Читает все изображения из INPUT_DIR, применяет к ним заданные
    последовательности преобразований и сохраняет в OUTPUT_DIR.
    """
    INPUT_DIR = "test_images"
    OUTPUT_DIR = "output"

    # --- НАСТРОЙКА ТЕСТОВ ---
    # Задайте здесь последовательности преобразований, которые хотите протестировать.
    # Каждая внутренняя последовательность будет применена к каждому изображению.
    pipelines_to_test = [
         ['hist_eq'],
         ['whitening'],
         ['whitening', 'hist_eq'],  # Сначала отбеливание, потом выравнивание
         ['hist_eq', 'whitening']  # Сначала выравнивание, потом отбеливание
    ]

    # Создаем выходную директорию, если она не существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Результаты будут сохранены в папку: {OUTPUT_DIR}")

    # Получаем список файлов из входной директории
    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    except FileNotFoundError:
        print(f"Ошибка: Папка '{INPUT_DIR}' не найдена. Пожалуйста, создайте ее и поместите туда изображения.")
        return

    if not image_files:
        print(f"В папке '{INPUT_DIR}' не найдено изображений.")
        return

    # Обрабатываем каждый файл
    for filename in image_files:
        input_path = os.path.join(INPUT_DIR, filename)

        # Загружаем изображение
        image = cv2.imread(input_path)
        if image is None:
            print(f"Не удалось прочитать файл: {filename}. Пропускаем.")
            continue

        print(f"\nОбработка файла: {filename}")

        # Применяем каждую заданную последовательность
        for pipeline in pipelines_to_test:
            processed_image = image.copy()

            print(f"...Применение последовательности: {', '.join(pipeline)}")
            for transform in pipeline:
                if transform == 'whitening':
                    processed_image = apply_whitening(processed_image)
                elif transform == 'hist_eq':
                    processed_image = apply_histogram_equalization(processed_image)

            # Генерируем имя для выходного файла
            base_name, extension = os.path.splitext(filename)
            transform_suffix = '_'.join(pipeline)
            output_filename = f"{base_name}_{transform_suffix}.png"  # Сохраняем в png для качества
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Сохраняем результат
            cv2.imwrite(output_path, processed_image)
            print(f"   -> Результат сохранен в: {output_path}")

    print("\nОбработка завершена.")


if __name__ == "__main__":
    main()