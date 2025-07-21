# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

# Убрали imageio, теперь только Pillow (PIL)
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ОБРАБОТКИ И СОХРАНЕНИЯ ИЗОБРАЖЕНИЯ ---
# Чтобы не дублировать код в блоках if/else, выносим всю логику сюда
def process_and_save_image(sess, phone, photo_filename, test_dir, res_sizes,
                           IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE,
                           enhanced_op, x_placeholder, iteration_num=None):
    """
    Загружает, обрабатывает и сохраняет одно изображение.
    """
    photo_path = os.path.join(test_dir, photo_filename)
    photo_name = photo_filename.rsplit(".", 1)[0]

    # Формируем имя для вывода в консоль
    log_prefix = f"Original model '{phone.replace('_orig', '')}'"
    if iteration_num:
        log_prefix = f"Iteration {iteration_num}"

    print(f"{log_prefix}, processing image {photo_filename}...")

    # 1. ЗАГРУЗКА И ПРИНУДИТЕЛЬНОЕ ИЗМЕНЕНИЕ РАЗМЕРА (как ты просил)
    # Открываем изображение с помощью Pillow
    image = Image.open(photo_path)

    # Получаем целевые размеры [высота, ширина]
    target_h, target_w = res_sizes[phone]

    # Приводим изображение к нужному размеру силой.
    # PIL.Image.resize принимает (ширина, высота), поэтому меняем местами
    # Image.Resampling.LANCZOS - это качественный фильтр для изменения размера
    resized_image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Конвертируем в numpy массив и нормализуем (от 0 до 1)
    image_np = np.float16(np.array(resized_image)) / 255.0

    # 2. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
    image_crop = image_np
    image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

    # 3. ПРОГОН ЧЕРЕЗ НЕЙРОСЕТЬ
    enhanced_2d = sess.run(enhanced_op, feed_dict={x_placeholder: image_crop_2d})
    enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # 4. СОХРАНЕНИЕ РЕЗУЛЬТАТА (твой код, который мы уже исправили)
    before_after = np.hstack((image_crop, enhanced_image))

    output_dir = "visual_results"
    os.makedirs(output_dir, exist_ok=True)

    # Конвертация в uint8 для сохранения
    if enhanced_image.dtype != np.uint8:
        enhanced_image_to_save = (np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8)
        before_after_to_save = (np.clip(before_after, 0, 1) * 255).astype(np.uint8)
    else:
        enhanced_image_to_save = enhanced_image
        before_after_to_save = before_after

    # Формируем имя выходного файла, добавляя итерацию, если она есть (ИСПРАВЛЕН БАГ)
    if iteration_num:
        base_filename = f"{phone}_iter_{iteration_num}_{photo_name}"
    else:
        base_filename = f"{phone}_{photo_name}"

    path_enhanced = os.path.join(output_dir, f"{base_filename}_enhanced.png")
    path_before_after = os.path.join(output_dir, f"{base_filename}_before_after.png")

    Image.fromarray(enhanced_image_to_save).save(path_enhanced)
    Image.fromarray(before_after_to_save).save(path_before_after)
    print(f" > Результаты сохранены в папку: {output_dir}")


# --- ОСНОВНОЙ СКРИПТ ---

tf.compat.v1.disable_v2_behavior()

# process command arguments
phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)

# get all available image resolutions
res_sizes = utils.get_resolutions()
res_sizes["iphone_orig"] = [1536, 2048] # <--- Добавляем твое разрешение для iphone_orig

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# create placeholders for input images
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:

    # Используем os.path.join для корректной работы на всех ОС
    test_dir = os.path.join(dped_dir, phone.replace("_orig", ""), "test_data/full_size_test_images/")
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

    if test_subset == "small":
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):
        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models_orig/" + phone)

        for photo in test_photos:
            # Вызываем нашу новую, чистую функцию
            process_and_save_image(sess, phone, photo, test_dir, res_sizes,
                                   IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE,
                                   enhanced, x_)

    else:
        # Получаем количество доступных моделей для итераций
        models_dir = "models/"
        num_saved_models = len([f for f in os.listdir(models_dir) if f.startswith(phone + "_iteration") and f.endswith(".ckpt.index")])

        if iteration == "all":
            iteration_list = np.arange(1, num_saved_models + 1) * 1000
        else:
            iteration_list = [int(iteration)]

        for i in iteration_list:
            # load pre-trained model
            model_path = os.path.join(models_dir, f"{phone}_iteration_{i}.ckpt")
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, model_path)

            for photo in test_photos:
                # Вызываем нашу функцию, передавая номер итерации для логов и имени файла
                process_and_save_image(sess, phone, photo, test_dir, res_sizes,
                                       IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE,
                                       enhanced, x_, iteration_num=i)