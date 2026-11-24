# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. В `C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts` лежат `train_eval.py`, `encoder.pt`, `contextualizer.pt` (BENDR v0.1-alpha); данные CHB-MIT (`chb01`, `chb02`, … с `*.edf` и `chbXX-summary.txt`) — в `...\3 семестр\data\chb-mit-scalp-eeg-database-1.0.0`. Другие пути — константы `SCRIPTS_DIR` / `DATA_DIR` в начале скрипта.
2. Зависимости: PyTorch с CUDA (https://pytorch.org), затем `pip install -r requirements.txt`.
3. `cd "C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts"`, затем по порядку:
   - модель учится? (зонд всех признаков + переобучение на 400 окнах, нужно AUC > 0.9): `python train_eval.py --sanity-overfit --freeze backbone`, затем `python train_eval.py --sanity-overfit --freeze partial --unfreeze-last 2`
   - быстрая проверка (только голова, 1 пациент): `python train_eval.py --test-patients chb01 --freeze backbone --run-name check`
   - partial: `python train_eval.py --freeze partial --unfreeze-last 2 --lr-head 1e-3 --lr-backbone 5e-5 --warmup 1 --unfreeze-epoch 2 --run-name partial2`
   - full: `python train_eval.py --freeze full --lr-head 1e-3 --lr-backbone 2e-5 --unfreeze-epoch 2 --run-name full2`
4. По умолчанию: вход как в BENDR (min-max окна в [-1,1] + канал SCALE, `--norm bendr`), признак `--feature auto` (выбирается зондом, таблица AUC в логе), голова BatchNorm→Linear (`--head mlp` — MLP), dropout 0, батч 64. Каждая эпоха печатает train/val loss, val AUC, Se/Sp при пороге Юдена, p mean±std. Все флаги: `python train_eval.py --help`.
5. Результаты: `scripts\results\<run-name>\summary.csv`, кэш окон — `scripts\results\cache` (пересчитывать не нужно). Готовые фолды переиспользуются: для нового прогона — новый `--run-name` или `--force`.
