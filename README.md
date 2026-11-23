# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. Python 3.10–3.13. Поставьте PyTorch под свою CUDA (команда с https://pytorch.org), затем: `pip install -r requirements.txt`
2. Запуск всего конвейера одной командой (сам скачает CHB-MIT ~43 ГБ и веса BENDR v0.1-alpha, построит кэш, обучит LOPO по chb01–chb23 и посчитает метрики):
   `python train_eval.py`
3. Варианты: `--freeze full|encoder|partial|backbone`, `--fractions 0.25 0.5 1.0`, `--scheme patient_wise`, `--patients chb01 chb02 chb03`, `--batch-size 64` (если не хватает видеопамяти). Все флаги: `python train_eval.py --help`.
4. Только подготовка данных: `python prepare_data.py`. Если GPU нет, обучение автоматически пойдёт на CPU.
5. Результаты: `results/<схема>_<freeze>/summary.csv` (метрики до/после пост-обработки), `metrics_all.csv`, `train.log`, чекпоинты `*/best.pt`.
