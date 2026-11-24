# BENDR: детекция приступов на CHB-MIT (НИР-3)

1. В `C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts` лежат `train_eval.py`, `encoder.pt`, `contextualizer.pt` (BENDR v0.1-alpha); данные CHB-MIT (`chb01`, `chb02`, … с `*.edf` и `chbXX-summary.txt`) — в `...\3 семестр\data\chb-mit-scalp-eeg-database-1.0.0`. Другие пути — константы `SCRIPTS_DIR` / `DATA_DIR` в начале скрипта.
2. Зависимости: PyTorch с CUDA (https://pytorch.org), затем `pip install -r requirements.txt`.
3. PowerShell: `conda activate bendr; $env:KMP_DUPLICATE_LIB_OK="TRUE"; cd "C:\Users\abuzo\Desktop\МАГИСТРАТУРА\МАГИСТЕРСКАЯ\3 семестр\scripts"`, затем по порядку:
   - модель учится? `python train_eval.py --sanity-overfit --freeze partial --unfreeze-last 2` — OK только если AUC ≥ 0.9 во ВСЕХ последних 5 эпохах, итоговый val loss < 0.6 и предсказания не схлопнулись
   - 1 пациент: `python train_eval.py --test-patients chb01 --freeze partial --unfreeze-last 2 --unfreeze-epoch 2 --run-name chb01_partial3`
   - partial: `python train_eval.py --freeze partial --unfreeze-last 2 --unfreeze-epoch 2 --run-name partial3`
   - full: `python train_eval.py --freeze full --unfreeze-epoch 2 --run-name full3`
4. По умолчанию: вход как в BENDR (min-max окна в [-1,1] + канал SCALE), признак `auto` (backbone — по зонду, дообучение — `ctx_mean`), голова BatchNorm→Linear со статистикой, пересчитываемой перед каждой оценкой, lr головы 1e-3 / магистрали 1e-5 (после разморозки — заново разогрев `--warmup` эпох), dropout 0, батч 64, bf16. После каждой эпохи строка `диагн.:` — величина признаков, расхождение bf16/fp32 и train/eval. Если bf16/fp32 Δ > 0.05: `--amp-scope encoder` (трансформер в fp32) или `--amp off`. Все флаги: `python train_eval.py --help`.
5. Результаты: `scripts\results\<run-name>\summary.csv`, кэш окон — `scripts\results\cache`. Готовые фолды переиспользуются: для нового прогона — новый `--run-name` или `--force`.
