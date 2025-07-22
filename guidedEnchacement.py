import os
import argparse
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
from PIL import Image

# Импортируем модули из вашего проекта
from models import resnet
import utils

# Отключаем поведение TensorFlow 2.x, так как проект использует v1 API
tf.disable_v2_behavior()


def run_dped_inference(image_low_res_np, model_name):
    """
    Выполняет инференс модели DPED для одного изображения.
    Эта функция инкапсулирует логику построения графа и запуска сессии из test_model.py,
    но делает ее динамической под размер входного изображения.

    Args:
        image_low_res_np (np.array): Изображение низкого разрешения (H, W, 3) в формате RGB, float32 [0, 1].
        model_name (str): Имя модели, например, 'iphone_orig'.

    Returns:
        np.array: Улучшенное изображение (H, W, 3) в формате RGB, float32 [0, 1].
    """
    # Сбрасываем граф по умолчанию, чтобы избежать конфликтов
    tf.reset_default_graph()

    # Получаем размеры входного изображения 'a'
    IMAGE_HEIGHT, IMAGE_WIDTH, _ = image_low_res_np.shape
    IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3

    # --- Создание графа TensorFlow (аналогично test_model.py) ---
    # Создаем плейсхолдеры для входного изображения
    x_ = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # Генерируем улучшенное изображение через модель resnet из вашего models.py
    enhanced_op = resnet(x_image)

    # GPU опции (можно выключить, если нужно)
    config = tf.ConfigProto(device_count={'GPU': 1})  # Используем GPU по умолчанию

    with tf.Session(config=config) as sess:
        # Загружаем предобученную модель
        model_path = os.path.join("models_orig", model_name)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print(f" > Модель '{model_name}' успешно загружена.")

        # Подготавливаем данные для подачи в модель
        # Модель ожидает плоский (1D) массив
        image_for_model = np.reshape(image_low_res_np, [1, IMAGE_SIZE])

        # Запускаем инференс
        enhanced_2d = sess.run(enhanced_op, feed_dict={x_: image_for_model})

        # Преобразуем результат обратно в формат изображения (H, W, 3)
        enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    return enhanced_image


def enhance_high_res(input_path, output_path, model_name, low_res_width=1024):
    """
    Улучшает изображение высокого разрешения, используя модель низкого разрешения
    и Guided Filter для апскейлинга.

    Args:
        input_path (str): Путь к входному изображению высокого разрешения.
        output_path (str): Путь для сохранения результата.
        model_name (str): Имя модели для DPED, например 'iphone_orig'.
        low_res_width (int): Целевая ширина для обработки в DPED.
    """
    print(f"Загрузка изображения: {input_path}")
    # Используем OpenCV для загрузки, т.к. он удобнее для конвертации форматов
    A_bgr = cv2.imread(input_path)
    if A_bgr is None:
        print(f"Ошибка: не удалось прочитать изображение {input_path}")
        return

    # Конвертируем в RGB и float [0, 1] для всех операций
    A = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = A.shape
    print(f"Оригинальное разрешение (A): {w}x{h}")

    # 1. Создаем 'a' - уменьшенную копию для модели
    scale_factor = low_res_width / w
    low_res_height = int(h * scale_factor)
    a = cv2.resize(A, (low_res_width, low_res_height), interpolation=cv2.INTER_AREA)
    print(f"Создание low-res версии (a) для модели: {low_res_width}x{low_res_height}")

    # 2. Получаем 'b' - улучшенную версию от DPED
    print("Запуск инференса DPED для получения 'b'...")
    # Наша новая функция использует существующую структуру проекта
    b = run_dped_inference(a, model_name)
    print("Инференс завершен.")

    # 3. Конвертируем все в CIE LAB для перцепционно корректной работы
    # OpenCV работает с float32 для Lab конвертации
    A_lab = cv2.cvtColor(A, cv2.COLOR_RGB2Lab)
    a_lab = cv2.cvtColor(a, cv2.COLOR_RGB2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_RGB2Lab)

    # 4. Вычисляем дельту (фильтр улучшения) в низком разрешении
    # delta = b - a
    delta_lab = b_lab - a_lab

    # 5. Guided Upsampling
    print("Выполнение Guided Upsampling дельты...")

    # Сначала апскейлим дельту до полного разрешения простым методом
    full_h, full_w = A.shape[:2]
    delta_lab_upscaled = cv2.resize(delta_lab, (full_w, full_h), interpolation=cv2.INTER_CUBIC)

    # Теперь применяем Guided Filter, используя L-канал оригинала как гайд
    # Это перенесет структуру с A на нашу дельту, предотвращая размытие
    L_A = A_lab[:, :, 0]  # Яркостный канал - наш гайд

    # Параметры для фильтра
    radius = 32  # Большой радиус для плавного результата
    eps = 1e-5  # Регуляризация

    # Применяем фильтр к каждому каналу дельты отдельно
    delta_L_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 0], radius, eps)
    delta_a_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 1], radius, eps)
    delta_b_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 2], radius, eps)

    # Собираем отфильтрованную дельту
    delta_high_lab = cv2.merge([delta_L_high, delta_a_high, delta_b_high])
    print("Guided Upsampling завершено.")

    # 6. Применяем полноразмерную дельту к оригиналу и собираем финальное изображение
    # B = A + delta_high
    B_lab = A_lab + delta_high_lab

    # Конвертируем обратно в RGB
    B_rgb = cv2.cvtColor(B_lab, cv2.COLOR_Lab2RGB)

    # 7. Сохраняем результат
    # Приводим к формату uint8 и конвертируем в BGR для сохранения в OpenCV
    B_to_save = np.clip(B_rgb * 255.0, 0, 255).astype(np.uint8)
    B_bgr_to_save = cv2.cvtColor(B_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, B_bgr_to_save)
    print(f"Результат успешно сохранен в: {output_path}")



if __name__ == '__main__':
    # Определяем пути
    INPUT_DIR = "dped/iphone/test_data/full_size_test_images/"
    BASE_OUTPUT_DIR = "visual_results"

    # Создаем парсер только для аргументов, которые мы хотим оставить
    parser = argparse.ArgumentParser(
        description="High-Resolution Image Enhancement for a folder of images.")
    parser.add_argument('--model', type=str, default='iphone_orig',
                        help="Model to use (e.g., 'iphone_orig', 'sony_orig'). Default: 'iphone_orig'.")
    parser.add_argument('--width', type=int, default=1024,
                        help="Width of the low-resolution image to be fed into the network. Default: 1024.")

    args = parser.parse_args()

    # --- Проверки перед запуском ---
    # Проверяем, существует ли модель
    model_path_check = os.path.join("models_orig", args.model + ".index")
    if not os.path.exists(model_path_check):
        print(f"Ошибка: Файл модели не найден по пути {model_path_check}")
        print(
            "Убедитесь, что вы указали правильное имя модели (например, 'iphone_orig') и файл находится в папке 'models_orig'.")
        exit()

    # Проверяем, существует ли папка с входными изображениями
    if not os.path.isdir(INPUT_DIR):
        print(f"Ошибка: Входная папка не найдена: {INPUT_DIR}")
        exit()

    # --- Создание папки для результатов ---
    run_num = 1
    # Ищем следующий свободный номер для папки run_x
    while os.path.exists(os.path.join(BASE_OUTPUT_DIR, f'run_{run_num}')):
        run_num += 1

    output_run_dir = os.path.join(BASE_OUTPUT_DIR, f'run_{run_num}')
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Результаты будут сохранены в папку: {output_run_dir}")

    # --- Получение списка изображений и их обработка ---
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"В папке {INPUT_DIR} не найдено изображений для обработки.")
        exit()

    total_images = len(image_files)
    print(f"Найдено {total_images} изображений для обработки.")

    for i, filename in enumerate(image_files):
        print("\n" + "=" * 50)
        print(f"Обработка изображения {i + 1}/{total_images}: {filename}")
        print("=" * 50)

        input_image_path = os.path.join(INPUT_DIR, filename)
        output_image_path = os.path.join(output_run_dir, filename)

        # Вызываем основную функцию для обработки одного изображения
        enhance_high_res(input_image_path, output_image_path, args.model, args.width)

    print("\nВсе изображения успешно обработаны!")