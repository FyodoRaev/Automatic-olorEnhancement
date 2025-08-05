# onnx_solution/convert_to_onnx.py
import os
import argparse
import tensorflow.compat.v1 as tf
import tf2onnx

# Импортируем модуль resnet из корневой папки проекта

from models import resnet

# Отключаем поведение TensorFlow 2.x
tf.disable_v2_behavior()


def convert_model(model_name, output_dir):
    """
    Загружает модель TensorFlow, "замораживает" граф
    и конвертирует ее в формат ONNX.
    """
    print(f"Начало конвертации модели '{model_name}'...")

    # --- 1. Создание графа TensorFlow ---
    tf.reset_default_graph()
    input_tensor = tf.placeholder(tf.float32, [None, None, None, 3], name='input')
    output_tensor = resnet(input_tensor)
    output_tensor = tf.identity(output_tensor, name='output')

    # --- 2. Загрузка весов в сессию ---
    model_path = os.path.join("models_orig", model_name)
    model_path_check = model_path + ".index"

    if not os.path.exists(model_path_check):
        print(f"Ошибка: Файл весов не найден: {model_path_check}")
        return

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("Веса TensorFlow успешно загружены.")

    # --- 3. Заморозка графа (ИСПРАВЛЕННАЯ ВЕРСИЯ) ---
    # Используем правильный импорт функции
    print("Заморозка графа...")
    output_node_name = output_tensor.name.split(':')[0]  # Получаем чистое имя узла, например 'output'

    # Используем tf.compat.v1.graph_util.convert_variables_to_constants
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        [output_node_name]
    )
    print("Граф успешно заморожен.")

    # --- 4. Конвертация ЗАМОРОЖЕННОГО графа в ONNX ---
    print("Конвертация в ONNX...")
    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        input_names=[input_tensor.name],
        output_names=[output_tensor.name],
        opset=13,
        output_path=None  # Мы сохраним файл вручную
    )

    # --- 5. Сохранение ONNX модели ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print("-" * 50)
    print(f"Модель успешно сконвертирована и сохранена в:")
    print(f"  {os.path.abspath(output_path)}")
    print("-" * 50)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Конвертер моделей DPED из TensorFlow в ONNX.")
    parser.add_argument('--model', type=str, default='sony_orig',
                        help="Имя модели для конвертации (например, 'iphone_orig', 'sony_orig').")
    parser.add_argument('--out_dir', type=str, default='onnx_models',
                        help="Папка для сохранения сконвертированных .onnx моделей.")

    args = parser.parse_args()

    # Создаем папку для ONNX моделей внутри onnx_solution
    output_directory = os.path.join(os.path.dirname(__file__), args.out_dir)

    convert_model(args.model, output_directory)