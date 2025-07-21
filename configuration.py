import torch
from torch import optim
from torch.utils.data import DataLoader
import os
from models.DeepBilateralNetCurves import DeepBilateralNetCurves
from models.DeepBilateralNetPointwiseNNGuide import DeepBilateralNetPointwiseNNGuide
from datasets.BaseDataset import BaseDataset
import utils

# Class choice parameters
model_class = 'Curves'
dataset_class = 'Base'
data_dir = 'data/debug'
pretrained_path = 'saved_runs/best/model.pth'

# Model parameters
lowres = [256, 256]
fullres = [1024, 1024]
spatial_bins = 16
luma_bins = 8
channel_multiplier = 1
guide_pts = 16

# Training parameters
n_epochs = 100
batch_size = 4
lr = 1e-4


def params():
    train_loader, test_loader = get_dataloaders()
    model = get_model()
    optimizer, scheduler = get_optimizer(model)
    return n_epochs, train_loader, test_loader, model, optimizer, scheduler


def get_dataloaders():
    if dataset_class == 'Base':
        train_set = BaseDataset(os.path.join(data_dir, 'train'), lowres, fullres, training=True)
        test_set = BaseDataset(os.path.join(data_dir, 'test'), lowres, fullres)
    else:
        raise NotImplementedError()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=2)
    return train_loader, test_loader


def get_model():
    if model_class == 'Curves':
        model = DeepBilateralNetCurves(lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts)
    elif model_class == 'NN':
        model = DeepBilateralNetPointwiseNNGuide(lowres, luma_bins, spatial_bins, channel_multiplier, guide_pts)
    else:
        raise NotImplementedError()
    if torch.cuda.is_available():
        model.cuda()
    if pretrained_path is not None:
        utils.load_model(model, pretrained_path)
    return model


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[n_epochs // 2 + i * 10 for i in range(10)],
                                               gamma=0.5)
    return optimizer, scheduler


# ====================================================================
# КОД ДЛЯ БЫСТРОЙ ПРОВЕРКИ ОДНОГО ИЗОБРАЖЕНИЯ
# Добавьте этот блок в конец вашего файла config.py
# ====================================================================

# Проверяем, что этот код запускается только если файл исполняется напрямую,
# а не импортируется другим скриптом.
if __name__ == '__main__':
    import torch
    from PIL import Image
    from torchvision import transforms

    # --- НАСТРОЙКИ ---
    INPUT_IMAGE_PATH = 'input/5.jpg'  # Укажите путь к вашему изображению
    OUTPUT_IMAGE_PATH = 'output/5.png'  # Имя файла для сохранения результата
    # -----------------

    print("1. Загрузка модели и весов...")

    # Получаем модель с уже правильно подобранными параметрами
    model = get_model()

    # Переводим модель в режим оценки (важно для отключения dropout и т.д.)
    model.eval()

    # Определяем, использовать CPU или GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Модель успешно загружена и работает на {device}.")

    # --- ПОДГОТОВКА ИЗОБРАЖЕНИЯ ---
    print(f"2. Подготовка изображения '{INPUT_IMAGE_PATH}'...")

    # Загружаем изображение с помощью библиотеки Pillow
    input_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')

    # Создаем трансформации для подготовки изображения
    # В модель подаются две версии: lowres и fullres
    transform_fullres = transforms.Compose([
        transforms.Resize(fullres),
        transforms.ToTensor()
    ])

    transform_lowres = transforms.Compose([
        transforms.Resize(lowres),
        transforms.ToTensor()
    ])

    # Применяем трансформации
    image_fullres = transform_fullres(input_image)
    image_lowres = transform_lowres(input_image)

    # Модель ожидает на вход "батч" (пачку) изображений.
    # Добавляем фиктивное первое измерение (batch size = 1)
    image_fullres = image_fullres.unsqueeze(0).to(device)
    image_lowres = image_lowres.unsqueeze(0).to(device)

    print("Изображение подготовлено.")

    # --- ОБРАБОТКА ИЗОБРАЖЕНИЯ (ИНФЕРЕНС) ---
    print("3. Обработка изображения моделью...")

    # Отключаем расчет градиентов для ускорения и экономии памяти
    with torch.no_grad():
        # Подаем обе версии изображения в модель
        prediction = model(image_lowres, image_fullres)

    print("Обработка завершена.")

    # --- СОХРАНЕНИЕ РЕЗУЛЬТАТА ---
    print(f"4. Сохранение результата в '{OUTPUT_IMAGE_PATH}'...")


    to_pil = transforms.ToPILImage()
    # Убираем фиктивное измерение батча и конвертируем
    # .squeeze(0) убирает batch-размерность, .cpu() перемещает на процессор
    output_image = to_pil(prediction.squeeze(0).cpu())

    # Сохраняем итоговое изображение
    output_image.save(OUTPUT_IMAGE_PATH)

    print("Готово! Проверьте файл output.png.")