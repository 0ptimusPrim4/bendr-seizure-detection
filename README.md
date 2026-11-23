# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. Положите `train_eval.py`, `encoder.pt` и `contextualizer.pt` (BENDR v0.1-alpha) в `C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts`.
2. Данные CHB-MIT (папки `chb01`, `chb02`, … с `*.edf` и `chbXX-summary.txt`) — в `C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\data\chb-mit-scalp-eeg-database-1.0.0`. Другие пути — в константах `SCRIPTS_DIR` / `DATA_DIR` в начале скрипта.
3. Зависимости: PyTorch с CUDA (команда с https://pytorch.org), затем `pip install -r requirements.txt`.
4. Запуск (cmd или PowerShell):
   `cd "C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts"`
   `python train_eval.py`
5. Флаги: `--freeze full|encoder|partial|backbone`, `--fractions 0.25 0.5 1.0`, `--scheme lopo|patient_wise`, `--patients chb01 chb02 chb03`, `--batch-size 64`, `--eval-only`; все — `python train_eval.py --help`. Без GPU скрипт сам переходит на CPU.
6. Результаты: `scripts\results\<схема>_<freeze>\summary.csv` (метрики до/после пост-обработки), кэш окон — `scripts\results\cache`.
