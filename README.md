# Predicția volatilității NYSE cu rețele neuronale

Proiect pentru *Rețele neuronale și tehnici de Deep Learning* (ASE). Scopul:
prezicerea **volatilității** acțiunilor de pe NYSE pe orizonturi multiple
(`1, 5, 10, 20` zile), comparând un model clasic (**GARCH**) cu rețele neuronale
recurente (**LSTM** și **Encoder-Decoder + Attention**), cu accent pe căutarea
iterativă a hiperparametrilor.

## Ideea pe scurt

- **Țintă:** volatilitatea realizată viitoare, estimată din OHLC cu estimatori
  *range-based* (implicit **Garman-Klass**; sunt implementați și Parkinson,
  Rogers-Satchell, Yang-Zhang, close-to-close).
- **Intrări (multivariate):** log-randament, log-range, log-volum, cei patru
  estimatori de volatilitate zilnică, ziua din săptămână (ciclic).
- **Scop:** un model **global** antrenat pe un coș de ~30 de simboluri lichide.
- **Split cronologic** (fără scurgere de informație): train ≤ 2018,
  validare 2019–2021, test > 2021 (configurabil în `nyse_vol/config.py`).

## Structura

```
nyse_vol/
  config.py                 # căi, simboluri, orizonturi, spații de căutare HPO
  data/
    sample_generator.py     # OHLCV sintetic în formatul real NYSE (rulare fără date reale)
    loader.py               # zip an -> CSV zilnice -> DataFrame long, cu curățare
    volatility.py           # Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, close-to-close
    features.py             # trăsături + ținte multi-orizont (log-volatilitate)
    dataset.py              # ferestre glisante, split cronologic, scalare, DataLoader
  models/
    garch.py                # baseline GARCH(p,q) walk-forward (lib. `arch`)
    lstm.py                 # LSTM/GRU stivuit, bidirecțional opțional
    seq2seq_attention.py    # Encoder-Decoder LSTM + atenție Bahdanau/Luong
  train/
    trainer.py              # buclă antrenare, early stopping, grad clipping
    hpo.py                  # random search (optimizatori, straturi, activări, regularizare)
  eval/
    metrics.py              # RMSE, MAE, QLIKE, R², acuratețe direcțională
    plots.py                # curbe loss, pred vs real, heatmap atenție, comparații
scripts/
  01_prepare_data.py        # date + trăsături (generează sintetic dacă lipsesc)
  02_train_garch.py         # baseline clasic
  03_train_nn.py            # antrenează LSTM și Attention
  04_run_hpo.py             # căutare hiperparametri
  05_evaluate_compare.py    # tabel + grafice comparative
tests/
  test_volatility.py        # teste numerice pentru estimatori
```

## Instalare

```bash
pip install -r requirements.txt
```

## Date

Codul citește zip-uri an cu an (`NYSE_2001.zip` … `NYSE_2026.zip`) cu fișiere
zilnice `NYSE_YYYYMMDD.csv` (`Symbol,Date,Open,High,Low,Close,Volume`). Setează
calea cu variabila de mediu `DATA_DIR`:

```bash
export DATA_DIR=/cale/catre/datele_NYSE
```

Dacă `DATA_DIR` nu conține zip-uri, pipeline-ul **generează automat date sintetice**
în același format, ca să poată rula imediat.

## Rulare end-to-end

```bash
python scripts/01_prepare_data.py            # date + trăsături (cache)
python scripts/02_train_garch.py             # baseline GARCH (--search pentru iterație p,q,dist)
python scripts/03_train_nn.py --epochs 30    # LSTM + Attention
python scripts/04_run_hpo.py --trials 10     # căutare hiperparametri (LSTM)
python scripts/05_evaluate_compare.py        # tabel + grafice comparative
```

Artefactele (modele, grafice, metrici) apar în `artifacts/`.

## Preprocesarea datelor

Pipeline-ul de preprocesare parcurge patru etape distincte, fiecare cu o
justificare clară. Codul se află în `nyse_vol/data/`.

### Etapa 1 — Curățarea datelor brute (`loader.py: _clean`)

Se aplică pe datele originale din ZIP-uri, **înainte** de orice imputare.

| Variabilă / situație | Ce se face | De ce |
|---|---|---|
| Valori non-numerice în OHLCV (`"-"`, `"N/A"`, string gol) | Conversie forțată la `NaN` cu `pd.to_numeric(errors="coerce")`, apoi rândul e eliminat | Nu poți imputa un preț de bursă fabricat; datele corupte sunt mai periculoase decât lipsa lor |
| Prețuri `Open/High/Low/Close ≤ 0` | Rândul e eliminat | Prețuri negative sau zero sunt imposibile pe piață și ar produce valori infinite sau NaN în transformările logaritmice (`log(C/O)`) |
| Zile nelichide: `High == Low` și `Volume == 0` | Rândul e eliminat | Estimatorii de volatilitate range-based (Parkinson, Garman-Klass) folosesc `log(H/L)`; dacă `H = L` obții `log(1) = 0`, adică volatilitate estimată zero — semnal fals, nu lipsă de date |

### Etapa 2 — Reindexare la calendarul bursier și forward-fill (`loader.py: _reindex_and_ffill`)

Se aplică **după** curățare, per simbol, față de toate zilele de tranzacționare
prezente în panel.

**Problema:** dacă un simbol lipsește complet dintr-un fișier zilnic (suspendare
temporară, eroare la provider), codul naiv nici nu detectează absența. Modelele
LSTM presupun că observațiile consecutive sunt la distanță egală în timp — o
fereastră de 60 „zile" poate conține de fapt 75 zile calendaristice, introducând
discontinuități temporale ascunse.

| Situație | Ce se face | De ce |
|---|---|---|
| Simbol absent 1–3 zile consecutive | Forward-fill OHLCV: ultimul preț cunoscut se propagă | Practică standard în quant finance — ultimul preț este cel mai bun estimat al valorii în absența tranzacțiilor. Produce rânduri cu `H=L=O=C` (volatilitate zero), semnal informativ: „nu s-a întâmplat nimic" |
| Simbol absent > 3 zile consecutive (`MAX_FFILL_DAYS`) | Rândurile rămân `NaN` și sunt eliminate | Un gap mai lung indică suspendare reală sau lipsă structurală de date; forward-fill-ul ar fabrica prețuri pentru prea mult timp și ar denatura estimatorii de volatilitate |

> **De ce nu se elimină zilele flat generate de forward-fill?**
> Spre deosebire de zilele flat din datele originale (care sunt date proaste),
> zilele flat din forward-fill sunt **intenționat informative**: rețeaua învață
> că volatilitate zero = zi fără informație nouă.

### Etapa 3 — Construirea features și țintelor (`features.py`)

| Variabilă | Transformare | De ce |
|---|---|---|
| `log_ret` = `log(Close_t / Close_{t-1})` | Log-randament zilnic | Log-randamentele sunt stationare și aproximativ normal distribuite, spre deosebire de prețuri (random walk). Prima zi din fiecare simbol → `NaN`, eliminat ulterior |
| `log_range` = `log(High / Low)` | Log-range intraday | Măsoară amplitudinea mișcării din zi; intrare directă în estimatorii de volatilitate. Pe zilele forward-fill: `log(1) = 0` — semnal valid de „zi inactivă" |
| `log_volume` = `log(max(Volume, 1))` | Log-volum | Volumul are distribuție puternic asimetrică (outlier-i masivi); log-ul comprimă scara și stabilizează varianța. `clip(lower=1)` evită `log(0)` |
| `vol_parkinson` | `sqrt(1/(4·ln2) · log(H/L)²)` | Estimator eficient al volatilității intraday folosind doar High/Low; mai precis decât close-to-close pe serii scurte |
| `vol_garman_klass` | Formulă OHLC completă | Utilizează toate cele 4 prețuri (O, H, L, C); cel mai precis estimator range-based clasic — ales ca **țintă principală** |
| `vol_rogers_satchell` | Formulă cu gap overnight | Robust la gap-urile de deschidere față de închiderea anterioară |
| `vol_close_to_close` | `sqrt(log(C_t/C_{t-1})²)` | Estimator naiv bazat doar pe randamentul zilnic; introdus ca baseline pentru comparație internă a features |
| `dow_sin`, `dow_cos` | `sin/cos(2π · DOW / 5)` | Codificare ciclică a zilei din săptămână (0=luni ... 4=vineri); permite rețelei să înțeleagă sezonalitatea săptămânală fără discontinuitate artificială la trecerea luni→vineri |
| `target_h{1,5,10,20}` | `log(sqrt(mean(var_t+1 ... var_t+h)))` | Volatilitate realizată viitoare, în scară logaritmică. Log-ul stabilizează varianța țintei (distribuție mai aproape de normală) și evită predicții negative la ieșirea din rețea |

**NaN introduse de feature engineering:**
- `log_ret` și `vol_close_to_close` → NaN pe prima zi (nu există `t-1`)
- `target_h20` → NaN pe ultimele 20 de zile din serie (nu există suficient viitor)
- Toate rândurile cu orice NaN în features sau ținte sunt **eliminate** — nu se poate imputa fără data leakage

### Etapa 4 — Scalare pentru antrenare (`dataset.py: build_splits`)

| Ce | Cum | De ce |
|---|---|---|
| **Features** (`X`) | `StandardScaler` — medie 0, deviație standard 1 | LSTM-urile sunt sensibile la scara inputurilor; scalarea accelerează convergența gradientului și evită saturarea funcțiilor de activare |
| **Ținte** (`y`) | Standardizare manuală: `(y - mean_train) / std_train` | MSE pe valori brute ar fi dominat de simbolurile cu volatilitate mare (ex: acțiuni tech volatile vs. utilities); standardizarea pune toate simbolurile pe același piedestal |
| **Fit exclusiv pe train** | `scaler.fit(X_train)`, apoi `transform` pe val și test | Evită data leakage: statisticile de scalare nu pot vedea date din validare sau test |

## Logging și depanare preprocesare

Scriptul `01_prepare_data.py` acceptă doi parametri opționali:

```bash
python scripts/01_prepare_data.py [--force] [--verbose]
```

| Flag | Efect |
|---|---|
| *(niciunul)* | Încarcă panel și features din cache (pickle); nu reprocessează nimic |
| `--force` | Ignoră cache-ul și reprocessează datele brute de la zero |
| `--verbose` | Activează log-uri la nivel DEBUG (per fișier ZIP, per simbol, per CSV zilnic) |
| `--force --verbose` | Reprocessează complet **și** afișează toate detaliile de preprocesare |

> **Important:** `--verbose` singur nu afișează log-urile din loader dacă există
> cache — în acel caz se vede doar mesajul `Panel incarcat din cache: ...`.
> Folosește întotdeauna `--force --verbose` când vrei să urmărești preprocesarea
> efectivă fișier cu fișier.

Exemplu de output cu `--force --verbose`:
```
10:23:41 | INFO  | _common        | Cache inexistent sau --force activ — procesez datele brute.
10:23:41 | INFO  | ...loader      | Procesez fisierul: NYSE_2020.zip
10:23:41 | INFO  | ...loader      |   Gasit 252 fisiere zilnice in arhiva (total intrari: 252)
10:23:41 | DEBUG | ...loader      |   Citesc: NYSE_20200102.csv (2020-01-02)
10:23:41 | DEBUG | ...loader      |   Citesc: NYSE_20200103.csv (2020-01-03)
           ...
10:23:42 | INFO  | ...loader      |   => 7560 randuri incarcate din NYSE_2020.zip
10:23:45 | WARNING | ...loader    |   [AIG] 2 zile lipsa din calendar: 2 umplute cu forward-fill, 0 eliminate
10:23:45 | INFO  | ...loader      |   Curatare: 156800 -> 156510 randuri (290 eliminate total)
```

## Note

- Pentru testare rapidă folosește `--epochs 2` și `--trials 3`.
- Datele reale și artefactele NU se comit (vezi `.gitignore`).
