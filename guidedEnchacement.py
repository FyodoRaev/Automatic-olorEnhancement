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


def load_model_and_graph(model_name):
    """
    Создает граф TensorFlow, загружает модель и возвращает необходимые для инференса объекты.
    Вызывается один раз перед обработкой изображений.

    Args:
        model_name (str): Имя модели для загрузки.

    Returns:
        tuple: (sess, input_tensor, output_tensor)
               - sess: Активная сессия TensorFlow с загруженной моделью.
               - input_tensor: Плейсхолдер для подачи изображений.
               - output_tensor: Тензор для получения результата.
    """
    # Сбрасываем граф по умолчанию, чтобы избежать конфликтов
    tf.reset_default_graph()

    # Создаем плейсхолдер с динамическими размерами [batch, height, width, channels]
    # Использование 'None' позволяет обрабатывать изображения любого размера без пересоздания графа.
    input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])

    # Генерируем улучшенное изображение через модель resnet
    output_tensor = resnet(input_tensor)

    # GPU опции
    config = tf.ConfigProto(device_count={'GPU': 1})
    sess = tf.Session(config=config)

    # Загружаем предобученную модель
    model_path = os.path.join("models_orig", model_name)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    return sess, input_tensor, output_tensor


def run_dped_inference(image_low_res_np, sess, input_tensor, output_tensor):
    """
    Выполняет инференс на УЖЕ загруженной модели DPED.

    Args:
        image_low_res_np (np.array): Изображение низкого разрешения (H, W, 3), float32 [0, 1].
        sess (tf.Session): Активная сессия TensorFlow.
        input_tensor (tf.Tensor): Плейсхолдер для подачи данных.
        output_tensor (tf.Tensor): Тензор для получения результата.

    Returns:
        np.array: Улучшенное изображение (H, W, 3), float32 [0, 1].
    """
    # Модель ожидает батч изображений, поэтому добавляем первую ось: (H, W, 3) -> (1, H, W, 3)
    image_for_model = np.expand_dims(image_low_res_np, axis=0)

    # Запускаем инференс
    enhanced_batch = sess.run(output_tensor, feed_dict={input_tensor: image_for_model})

    # Убираем лишнюю ось батча, возвращая изображение (1, H, W, 3) -> (H, W, 3)
    enhanced_image = np.squeeze(enhanced_batch, axis=0)

    return enhanced_image


def enhance_high_res(input_path, output_path, model_name, low_res_width, sess, input_tensor, output_tensor):
    """
    Улучшает изображение высокого разрешения, используя модель низкого разрешения
    и Guided Filter для апскейлинга.
    """
    print(f"Загрузка изображения: {input_path}")
    A_bgr = cv2.imread(input_path)
    if A_bgr is None:
        print(f"Ошибка: не удалось прочитать изображение {input_path}")
        return

    A = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = A.shape
    print(f"Оригинальное разрешение (A): {w}x{h}")

    scale_factor = low_res_width / w
    low_res_height = int(h * scale_factor)
    a = cv2.resize(A, (low_res_width, low_res_height), interpolation=cv2.INTER_AREA)
    print(f"Создание low-res версии (a) для модели: {low_res_width}x{low_res_height}")

    print("Запуск инференса DPED для получения 'b'...")
    # Вызываем новую, быструю функцию инференса
    b = run_dped_inference(a, sess, input_tensor, output_tensor)
    print("Инференс завершен.")

    A_lab = cv2.cvtColor(A, cv2.COLOR_RGB2Lab)
    a_lab = cv2.cvtColor(a, cv2.COLOR_RGB2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_RGB2Lab)

    delta_lab = b_lab - a_lab

    print("Выполнение Guided Upsampling дельты...")
    full_h, full_w = A.shape[:2]
    delta_lab_upscaled = cv2.resize(delta_lab, (full_w, full_h), interpolation=cv2.INTER_CUBIC)

    L_A = A_lab[:, :, 0]
    radius = 32
    eps = 1e-5

    delta_L_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 0], radius, eps)
    delta_a_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 1], radius, eps)
    delta_b_high = cv2.ximgproc.guidedFilter(L_A, delta_lab_upscaled[:, :, 2], radius, eps)

    delta_high_lab = cv2.merge([delta_L_high, delta_a_high, delta_b_high])
    print("Guided Upsampling завершено.")

    B_lab = A_lab + delta_high_lab
    B_rgb = cv2.cvtColor(B_lab, cv2.COLOR_Lab2RGB)

    B_to_save = np.clip(B_rgb * 255.0, 0, 255).astype(np.uint8)
    B_bgr_to_save = cv2.cvtColor(B_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, B_bgr_to_save)
    print(f"Результат успешно сохранен в: {output_path}")


# ==============================================================================
#                 ОСНОВНОЙ БЛОК ИСПОЛНЕНИЯ
# ==============================================================================
if __name__ == '__main__':
    INPUT_DIR = "dped/iphone/test_data/full_size_test_images/"
    BASE_OUTPUT_DIR = "visual_results"

    parser = argparse.ArgumentParser(
        description="High-Resolution Image Enhancement for a folder of images.")
    parser.add_argument('--model', type=str, default='iphone_orig',
                        help="Model to use (e.g., 'iphone_orig', 'sony_orig'). Default: 'iphone_orig'.")
    parser.add_argument('--width', type=int, default=1024,
                        help="Width of the low-resolution image to be fed into the network. Default: 1024.")

    args = parser.parse_args()

    model_path_check = os.path.join("models_orig", args.model + ".index")
    if not os.path.exists(model_path_check):
        print(f"Ошибка: Файл модели не найден по пути {model_path_check}")
        exit()

    if not os.path.isdir(INPUT_DIR):
        print(f"Ошибка: Входная папка не найдена: {INPUT_DIR}")
        exit()

    # --- ЗАГРУЖАЕМ МОДЕЛЬ ОДИН РАЗ ---
    print(f"Загрузка модели '{args.model}' в память... Это может занять некоторое время.")
    sess, input_tensor, output_tensor = load_model_and_graph(args.model)
    print("Модель успешно загружена и готова к работе.")
    # ------------------------------------

    run_num = 1
    while os.path.exists(os.path.join(BASE_OUTPUT_DIR, f'run_{run_num}')):
        run_num += 1

    output_run_dir = os.path.join(BASE_OUTPUT_DIR, f'run_{run_num}')
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Результаты будут сохранены в папку: {output_run_dir}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"В папке {INPUT_DIR} не найдено изображений для обработки.")
        exit()

    total_images = len(image_files)
    print(f"Найдено {total_images} изображений для обработки.")

    try:
        for i, filename in enumerate(image_files):
            print("\n" + "=" * 50)
            print(f"Обработка изображения {i + 1}/{total_images}: {filename}")
            print("=" * 50)

            input_image_path = os.path.join(INPUT_DIR, filename)
            output_image_path = os.path.join(output_run_dir, filename)

            # Вызываем основную функцию, передавая ей уже загруженную сессию и тензоры
            enhance_high_res(input_image_path, output_image_path, args.model, args.width,
                             sess, input_tensor, output_tensor)
    finally:
        # --- ЗАКРЫВАЕМ СЕССИЮ ПОСЛЕ ЗАВЕРШЕНИЯ РАБОТЫ ---
        print("\nОбработка завершена. Закрытие сессии TensorFlow.")
        sess.close()
        # ---------------------------------------------

    print("Все изображения успешно обработаны!")