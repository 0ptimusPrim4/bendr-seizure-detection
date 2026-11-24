# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. В `C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts` лежат `train_eval.py`, `encoder.pt`, `contextualizer.pt` (BENDR v0.1-alpha); данные CHB-MIT (`chb01`, `chb02`, … с `*.edf` и `chbXX-summary.txt`) — в `...\3 семестр\data\chb-mit-scalp-eeg-database-1.0.0`. Другие пути — константы `SCRIPTS_DIR` / `DATA_DIR` в начале скрипта.
2. Зависимости: PyTorch с CUDA (https://pytorch.org), затем `pip install -r requirements.txt`.
3. `cd "C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts"`, затем по порядку:
   - быстрая проверка (только голова, 1 пациент в тесте): `python train_eval.py --test-patients chb01 --freeze backbone --run-name check`
   - модель учится? (train AUC > 0.9 на ~400 окнах): `python train_eval.py --sanity-overfit --freeze partial --unfreeze-last 2`
   - partial: `python train_eval.py --freeze partial --unfreeze-last 2 --lr-head 1e-3 --lr-backbone 1e-5 --warmup 1 --unfreeze-epoch 2`
   - full: `python train_eval.py --freeze full --lr-head 1e-3 --lr-backbone 1e-5 --unfreeze-epoch 2`
4. Каждая эпоха печатает train/val loss, val AUC, Se/Sp при пороге Юдена, p mean±std; при постоянных предсказаниях или NaN — предупреждение. AMP: `--amp auto|bf16|fp16|off` (auto = bf16, если поддерживается). Все флаги: `python train_eval.py --help`.
5. Результаты: `scripts\results\<схема>_<freeze>\summary.csv`, кэш окон — `scripts\results\cache`. Старые предсказания переиспользуются: для нового прогона — `--run-name` или `--force`.
