
https://arxiv.org/pdf/1707.02880\
https://github.com/google/hdrnet
Это отвратительная реализация на доисторическом тензорфлоу. НО! Там есть претрейнд модели! А это важно 
Я попытался поставить эту модель, попробовал разные версии python, но продолжать разбираться в зарослях зависимостей старющего проекта - это ужас. 

Просто есть вот такой репозиторий https://github.com/creotiv/hdrnet-pytorch, но у него есть страшная проблема...

> there is no pretrained models, as there still problems with network training (no one knows why)


**EricElmoznino/deep_bilateral_network (PyTorch)** – развёрнутый PyTorch-код для «черного ящика» HDRNet [github.com](https://github.com/EricElmoznino/deep_bilateral_network#:~:text=,preferably%2C%20but%20not%20necessary). Требует PyTorch ≥0.4, Python 3.6+ и ряд библиотек (`tensorboardX`, OpenCV, scikit-video)[github.com](https://github.com/EricElmoznino/deep_bilateral_network#:~:text=,preferably%2C%20but%20not%20necessary). Содержится C++-слой для операции билинейной сетки: его нужно собрать командой

