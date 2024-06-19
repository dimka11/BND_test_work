# Запуск

прописать путь к видео, путь к видео с результатом детекции и название весов в config.yaml
выполнить python main.py в папке с проектом

# Описание решение

Поскольку, документе не указано необходимость обучения своей модели, ни датасета, ни рекомендаций к использованию
какого-то готового датасета, решено было использовать предобученные веса YOLOv5, т.к. имеется опыт с использованием этой
библиотеки, библиотека подддерживает работу с видео "из коробки", версия yolo не принципиальна (в более новых версиях 
такой-же интерфейс использования)

В коде сделал фильтрацию по классам ,чтобы отображалась только детекция класса Person (как описано в задании)

# Выводы

Т.к. была использована предобученная модель без дообучения, точность не очень хорошая, т.к. использование моделей без 
дообучения редко, когда дают хороший результат, плюс т.к. отсутствует датасет, то мы не можем оценить метрики решения.

Решение можно улучшить, если -
* дообучить модель на своем датасете
* использовать модель большего размера (large, x-large)
* подобрать гиперпараметры инференса (можно использовать Optuna либо другие библиотеки поиска гиперпараметров)
* использовать свой инференс модели (например какой-то другой NMS, проход окном по картинке и склейка результатов) - 
* врядли это бы дало улучшение на данном конкретном видео, но попробовать можно
* использовать другие библиотеки детекции (разные модели из mmdetection) с обучением на своем датасете - но это уже 
* после того, как все вышеперечисленное было проверено.