# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. Зависимости: PyTorch под свою CUDA (https://pytorch.org), затем `pip install -r requirements.txt`.
2. Данные: папки `chb01 … chb24` из CHB-MIT (`*.edf` + `chbXX-summary.txt`) положите в `data/chbmit/`.
3. Веса BENDR v0.1-alpha: `weights/encoder.pt` и `weights/contextualizer.pt`.
4. Запуск из папки репозитория (обучение LOPO + оценка + пост-обработка): `python train_eval.py`
5. Флаги: `--freeze full|encoder|partial|backbone`, `--fractions 0.25 0.5 1.0`, `--scheme lopo|patient_wise`, `--balancing sampler|class_weights`, `--patients chb01 chb02 …`, `--batch-size 64`, `--eval-only`; пути — `--data-dir`, `--encoder`, `--contextualizer`, `--out`. Все флаги: `python train_eval.py --help`.
6. Без GPU скрипт сам переходит на CPU. Результаты: `results/<схема>_<freeze>/summary.csv` (метрики до/после пост-обработки), кэш окон — `results/cache/`.
