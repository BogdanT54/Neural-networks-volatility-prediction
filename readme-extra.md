# Predictia Volatilitatii NYSE cu Retele Neuronale

Proiect de predictie a volatilitatii zilnice a actiunilor NYSE folosind modele LSTM si Encoder-Decoder cu Atentie, comparate cu baseline-urile clasice GARCH si modelul Naiv (persistence).

Datele acopera perioada **2001–2026** pentru **30 de simboluri lichide** NYSE. Splitul cronologic separa strict antrenarea, validarea si testul pentru a evita orice scurgere de informatii din viitor.

---

## Pe scurt

Proiectul nu incearca sa estimeze pretul exact al unei actiuni de maine. In schimb,
incearca sa prezica **volatilitatea**, adica nivelul de miscare sau instabilitate al
pretului. Practic, intrebarea este:

> Pe baza ultimelor 60 de zile de tranzactionare, cat de agitata va fi actiunea in
> urmatoarea zi, urmatoarea saptamana sau urmatoarea luna?

Aceasta alegere este mai potrivita pentru un proiect de serii de timp financiare,
deoarece volatilitatea are de obicei persistenta: perioadele agitate tind sa fie
urmate de alte perioade agitate, iar perioadele calme tind sa ramana calme o vreme.

Fluxul complet este:

```
CSV-uri NYSE zilnice
   → curatare si validare date
   → calcul indicatori financiari
   → ferestre de 60 de zile
   → antrenare LSTM / Attention
   → comparatie cu GARCH si Naiv
   → grafice + tabele de metrici
```

---

## Cuprins

1. [Obiectiv](#1-obiectiv)
2. [Date si simboluri](#2-date-si-simboluri)
3. [Structura proiectului](#3-structura-proiectului)
4. [Pipeline pas cu pas](#4-pipeline-pas-cu-pas)
5. [Features si tinte](#5-features-si-tinte)
6. [Arhitecturi de modele](#6-arhitecturi-de-modele)
7. [Antrenare si HPO](#7-antrenare-si-hpo)
8. [Evaluare si metrici](#8-evaluare-si-metrici)
9. [Grafice generate](#9-grafice-generate)
10. [Cum se ruleaza](#10-cum-se-ruleaza)
11. [Configurare](#11-configurare)
12. [Cerinte](#12-cerinte)
13. [Cum se citeste proiectul](#13-cum-se-citeste-proiectul)

---

## 1. Obiectiv

Proiectul prezice **volatilitatea viitoare** a actiunilor NYSE pe patru orizonturi:

| Orizont | Semnificatie                                                       |
| ------- | ------------------------------------------------------------------ |
| h = 1   | volatilitatea medie realizata maine                                |
| h = 5   | volatilitatea medie pe urmatoarele 5 zile bursiere (~1 saptamana)  |
| h = 10  | volatilitatea medie pe urmatoarele 10 zile bursiere (~2 saptamani) |
| h = 20  | volatilitatea medie pe urmatoarele 20 de zile bursiere (~1 luna)   |

**Tinta**: estimatorul Garman-Klass al volatilitatii zilnice (mai precis decat close-to-close, utilizeaza preturile OHLC complete).

**Motivatie practica**: predictia volatilitatii este centrala in managementul riscului, evaluarea optiunilor (formula Black-Scholes) si alocarea de portofolii (modele Markowitz).

### Interpretare intuitiva

Daca am prezice pretul, o tinta ar putea fi `Close` de maine. In acest proiect,
tinta este diferita: vrem sa estimam **marimea miscarii**, nu directia exacta a
pretului. Doua actiuni pot avea acelasi randament final intr-o zi, dar una poate
fi mult mai volatila daca a avut miscari mari intre `High` si `Low`.

De aceea folosim preturile:

- `Open`: de unde a pornit ziua;
- `High`: cel mai mare pret atins;
- `Low`: cel mai mic pret atins;
- `Close`: unde s-a inchis ziua.

Aceste patru valori spun mai mult despre agitatia pietei decat doar pretul de
inchidere.

---

## 2. Date si simboluri

### Simboluri NYSE

30 de actiuni lichide, prezente continuu pe NYSE din 2001:

```
AA   ABT  AEE  AEP  AES  AFL  AIG  AXP  BA   BAC
C    CAT  CL   CVX  DD   DIS  DOW  DUK  EMR  F
GE   GS   HD   IBM  JNJ  JPM  KO   MCD  MMM  MO
```

Lista poate fi modificata prin variabila de mediu `SYMBOLS`:

```bash
export SYMBOLS="JPM,GS,BAC,C,AXP"
```

### Format date brute

Fisiere CSV anuale in `data_raw/`, cu denumire `NYSE_AAAA.csv` (ex: `NYSE_2001.csv`), continand coloanele: `Symbol, Date, Open, High, Low, Close, Volume`.

In implementarea curenta, loader-ul accepta in special structura cu fisiere zilnice
grupate pe ani:

```
data_raw/
├── NYSE_2001/
│   ├── NYSE_20010102.csv
│   ├── NYSE_20010103.csv
│   └── ...
├── NYSE_2002/
│   └── ...
└── ...
```

Este acceptata si varianta cu arhive ZIP anuale:

```
data_raw/
├── NYSE_2001.zip
├── NYSE_2002.zip
└── ...
```

Fiecare CSV zilnic are aceleasi coloane standard: `Symbol, Date, Open, High, Low,
Close, Volume`. Codul citeste data direct din numele fisierului de tip
`NYSE_YYYYMMDD.csv`, pentru a evita problemele produse de formate diferite ale
coloanei `Date`.

### Split cronologic

```
Train:      2001-01-01  →  2018-12-31   (~18 ani)
Validare:   2019-01-01  →  2021-12-31   (~3 ani)
Test:       2022-01-01  →  2026-xx-xx   (~4 ani, set "secret" la evaluare)
```

Splitul este **strict cronologic** — nicio informatie din viitor nu patrunde in trecut (no look-ahead bias).

### De ce folosim doar o parte din simboluri?

Setul NYSE contine foarte multe companii, dar nu toate sunt la fel de utile pentru
antrenare. Unele pot avea istoric incomplet, perioade lungi fara tranzactionare sau
volume foarte mici. Pentru un prim model stabil, proiectul foloseste 30 de simboluri
lichide si prezente pe termen lung.

Aceasta alegere ajuta la:

- reducerea erorilor cauzate de lipsuri in date;
- obtinerea unor serii temporale lungi;
- antrenarea unui model global care vede mai multe exemple;
- compararea mai usoara a rezultatelor intre actiuni.

---

## 3. Structura proiectului

```
Neural-networks-volatility-prediction/
│
├── data_raw/                        # Date brute NYSE (CSV anual)
│   ├── NYSE_2001.csv
│   ├── NYSE_2002.csv
│   └── ...
│
├── nyse_vol/                        # Pachetul principal Python
│   ├── config.py                    # Configurare centrala (cai, simboluri, hiperparametri)
│   ├── data/
│   │   ├── features.py              # Calcul features si tinte
│   │   ├── volatility.py            # Estimatori de volatilitate (Parkinson, GK, RS, CtC, YZ)
│   │   └── dataset.py               # Ferestre glisante, splits, scalare, DataLoaders
│   ├── models/
│   │   ├── lstm.py                  # Model LSTM (stacked, optional bidirectional)
│   │   ├── seq2seq_attention.py     # Encoder-Decoder + Atentie (Bahdanau / Luong)
│   │   └── garch.py                 # Baseline GARCH + teste de stationaritate
│   ├── train/
│   │   ├── trainer.py               # Bucla de antrenare (early stopping, LR scheduler)
│   │   └── hpo.py                   # Random search pentru hiperparametri
│   └── eval/
│       ├── metrics.py               # RMSE, MAE, R², QLIKE, Dir.Acc
│       └── plots.py                 # Toate graficele proiectului
│
├── scripts/                         # Scripturi executabile (pasii 01-06)
│   ├── _common.py                   # Utilitare comune (cache Bronze->Silver->Gold)
│   ├── 01_prepare_data.py           # Pas 1: preprocesare si feature engineering
│   ├── 02_train_garch.py            # Pas 2: baseline GARCH walk-forward
│   ├── 03_train_nn.py               # Pas 3: antrenare LSTM si Attention (referinta)
│   ├── 04_run_hpo.py                # Pas 4: optimizare hiperparametri (random search)
│   ├── 05_evaluate_compare.py       # Pas 5: evaluare si comparare toate modelele
│   └── 06_select_and_plot.py        # Pas 6: grafice detaliate per stock ales
│
└── artifacts/                       # Generate automat
    ├── processed/                   # Cache-uri Bronze/Silver/Gold (Parquet)
    ├── models/                      # Checkpointuri .pt
    ├── metrics/                     # CSV-uri cu metrici si config HPO
    └── plots/
        ├── antrenare/               # Loss curves, attention heatmap
        ├── comparatii/              # Bar charts RMSE/QLIKE comparative
        ├── predictii/               # Pred vs real per model+simbol
        └── stocuri/                 # Grafice detaliate per stock ales
```

---

## 4. Pipeline pas cu pas

### Arhitectura Medallion (Bronze -> Silver -> Gold)

Procesarea datelor urmeaza un model in trei straturi cu validare la fiecare tranzitie:

```
[Bronze]  Date brute CSV   -->  DataFrame complet, toate simbolurile
    |     validare: coloane prezente, tipuri, valori nule
[Silver]  Date curatate    -->  interpolare max 2 zile consecutive lipsa, simbol activ
    |     validare: min 250 observatii per simbol
[Gold]    Features + tinte -->  9 features + 4 tinte, gata pentru modelele NN
```

Cache-ul este salvat ca fisiere Parquet in `artifacts/processed/`. Daca simbolurile nu s-au schimbat, etapele de preprocesare sunt sarite la reluari.

### Ce inseamna Bronze, Silver si Gold?

Acesti termeni sunt doar o metoda de organizare a datelor:

- **Bronze**: datele sunt citite, dar sunt aproape brute;
- **Silver**: datele sunt curatate, validate si aliniate;
- **Gold**: datele sunt gata de folosit de model, cu features si tinte calculate.

Separarea este utila deoarece putem verifica fiecare etapa separat. Daca apar valori
ciudate in rezultate, putem vedea daca problema vine din datele brute, din curatare
sau din calculul indicatorilor.

### Pas 1 — Preprocesare si feature engineering (`01_prepare_data.py`)

- Incarca toate fisierele CSV din `data_raw/`
- Valideaza si curata datele (Silver): valori lipsa, interpolare max 2 zile consecutive
- Calculeaza cei 9 features si cele 4 tinte (Gold)
- Optional (`--export`): exporta `features_all.csv`, `panel_silver.csv`, statistici

In termeni simpli, acest pas transforma fisierele CSV intr-un tabel curat si apoi
adauga coloane noi care pot fi intelese mai usor de un model neural. De exemplu,
pretul brut nu este folosit direct ca singura informatie; din el se calculeaza
randamente, intervale intraday si estimatori de volatilitate.

### Pas 2 — Baseline GARCH (`02_train_garch.py`)

- Ruleaza **teste de stationaritate** pe log-randamentele fiecarui simbol:
  - ADF (H0 = nestationar; p < 0.05 stationar)
  - KPSS (H0 = stationar; p >= 0.05 stationar)
  - Phillips-Perron (H0 = nestationar; p < 0.05 stationar)
- Testeaza **8 configuratii GARCH** (4 ordine x 2 distributii):
  - Ordine: (1,1), (1,2), (2,1), (2,2)
  - Distributii: normal, Student-t
  - Compara pe RMSE, MAE, QLIKE (out-of-sample) si AIC, BIC (in-sample)
- **Evaluare walk-forward** pe setul de test: refit la fiecare 21 de zile bursiere
  pe fereastra expanding; intre refitturi, varianta conditionala este actualizata
  zilnic cu randamentul observat: `h_{t+1} = omega + alpha * r_t^2 + beta * h_t`
- Salveaza cel mai bun model ca `artifacts/metrics/garch_forecasts.pkl`

### Pas 3 — Antrenare NN de referinta (`03_train_nn.py`)

- Antreneaza LSTM si/sau Encoder-Decoder+Atentie cu **hiperparametri impliciți**
- Salveaza `artifacts/models/lstm.pt` si `artifacts/models/attention.pt`
- Genereaza loss curves si attention heatmap in `plots/antrenare/`

### Pas 4 — Optimizare hiperparametri (`04_run_hpo.py`)

- **Random search** pe spatiul de cautare definit in `config.py`
- Evalueaza fiecare trial pe setul de **validare** (testul ramane secret)
- Retreneaza modelul ales cu numarul complet de epoci
- Salveaza `lstm_best_hpo.pt` / `attention_best_hpo.pt` si config JSON
- Ruleaza separat pentru fiecare model: `--model lstm` si `--model attention`

### Pas 5 — Evaluare comparativa (`05_evaluate_compare.py`)

- Incarca toate modelele disponibile (prioritate HPO > referinta)
- Calculeaza RMSE, MAE, R², QLIKE, Dir.Acc pe setul de test
- Genereaza bar charts comparative si grafice pred vs. real
- Salveaza `artifacts/metrics/comparison.csv`

Acesta este pasul principal pentru concluzii. Daca vrem sa stim care model este
mai bun, ne uitam aici: in tabelul `comparison.csv` si in graficele comparative.
Modelele sunt evaluate pe date pe care nu le-au folosit la antrenare.

### Pas 6 — Selectie stock-uri si grafice finale (`06_select_and_plot.py`)

- Permite alegerea unui subset de simboluri (interactiv sau `--symbols`)
- Selecteaza automat cel mai bun model dupa RMSE h=1 pe tot setul de test
- Genereaza per stock ales:
  - Subplot separat per model (Real vs. Prezis la h=1, cu RMSE in titlu)
  - 4 subploturi (h=1,5,10,20) pentru cel mai bun model dupa RMSE
  - Bar chart cu metrici RMSE/MAE/R²/QLIKE per model
- Salveaza `artifacts/metrics/results_selected_stocks.csv`

---

## 5. Features si tinte

### Cele 9 features de intrare

Fiecare fereastra de intrare contine **60 de zile x 9 features**, standardizate cu
`StandardScaler` fit-uit exclusiv pe datele de antrenare.

| Feature               | Formula                                             | Semnificatie                                  |
| --------------------- | --------------------------------------------------- | --------------------------------------------- |
| `log_ret`             | `log(Close_t / Close_{t-1})`                        | Randamentul logaritmic zilnic                 |
| `log_range`           | `log(High / Low)`                                   | Amplitudinea intraday (proxy de volatilitate) |
| `log_volume`          | `log(Volume)`                                       | Volum comprimat logaritmic                    |
| `vol_parkinson`       | `sqrt( [log(H/L)]^2 / (4*ln2) )`                    | Volatilitate Parkinson (1980), range-based    |
| `vol_garman_klass`    | `sqrt( 0.5*[log(H/L)]^2 - (2*ln2-1)*[log(C/O)]^2 )` | Garman-Klass (1980), OHLC                     |
| `vol_rogers_satchell` | `sqrt( log(H/O)*log(H/C) + log(L/O)*log(L/C) )`     | Rogers-Satchell (1991), robust la drift       |
| `vol_close_to_close`  | `sqrt( [log(C_t/C_{t-1})]^2 )`                      | Volatilitate simpla, randament la patrat      |
| `dow_sin`             | `sin(2*pi * zi_saptamana / 5)`                      | Sezonalitate intraweek (codare ciclica)       |
| `dow_cos`             | `cos(2*pi * zi_saptamana / 5)`                      | Sezonalitate intraweek (codare ciclica)       |

**De ce mai multi estimatori de volatilitate ca features?** Fiecare capteaza informatii
diferite: Parkinson foloseste doar H/L, Rogers-Satchell este robust la driftul overnight,
Garman-Klass combina toate preturile. Impreuna ofera modelului o descriere
multidimensionala a regimului curent de volatilitate.

**De ce codare ciclica (sin/cos) pentru ziua saptamanii?** O codare liniara (luni=0,
marti=1, ..., vineri=4) nu ar captura ca vineri si luni sunt "vecine" in ciclul
saptamanal. Sin/cos mapeaza ziua pe un cerc, pastrandu-se continuitatea.

### De ce nu folosim doar pretul `Close`?

Pretul de inchidere este important, dar pierde multa informatie. Intr-o zi, o actiune
poate sa deschida la 100, sa urce la 110, sa coboare la 95 si sa inchida tot la 100.
Daca ne uitam doar la `Close`, pare ca nu s-a intamplat nimic. Dar, in realitate,
ziua a fost foarte volatila.

Din acest motiv, proiectul foloseste:

- randamente, pentru schimbarea de la o zi la alta;
- `High` si `Low`, pentru miscarea din interiorul zilei;
- volum, pentru intensitatea tranzactionarii;
- estimatori de volatilitate, pentru a transforma preturile in semnale mai utile.

### Cele 4 tinte de iesire

Modelul prezice simultan **4 valori** per zi, corespunzand celor 4 orizonturi.
Formula din `features.py`:

```python
daily_var = garman_klass(Open, High, Low, Close)   # varianta zilnica

fwd_var = (daily_var.shift(-1)        # decaleaza cu 1 zi in viitor
           .rolling(hz).mean()        # media variantei pe hz zile consecutive
           .shift(-(hz - 1)))         # aliniaza fereastra sa inceapa de la t+1

target_h{hz}(t) = log( sqrt(fwd_var(t)) )
```

**Concret, pentru o zi `t`:**

| Coloana      | Ce reprezinta                                     | Formulare detaliata                         |
| ------------ | ------------------------------------------------- | ------------------------------------------- |
| `target_h1`  | log-volatilitatea realizata **maine** (ziua t+1)  | `log(sigma_GK(t+1))`                        |
| `target_h5`  | log-volatilitatea medie pe **urmatoarele 5 zile** | `log( sqrt( mean(sigma^2_GK(t+1..t+5)) ) )` |
| `target_h10` | media pe **t+1...t+10**                           | similar                                     |
| `target_h20` | media pe **t+1...t+20**                           | similar                                     |

**De ce `log`?** Volatilitatea bruta are o distributie log-normala (spikes rare dar
extreme). In spatiul logaritmic distributia este mai apropiata de normala, gradientii
sunt mai stabili si MSE-ul converge mai bine. La evaluare se aplica `exp()` pentru
revenire la scala originala (unitati: volatilitate zilnica absoluta, tipic 0.01-0.05).

### Exemplu de interpretare a unei tinte

Daca pentru un simbol avem `target_h5` intr-o anumita zi, acea valoare nu inseamna
pretul peste 5 zile. Inseamna volatilitatea medie realizata in urmatoarele 5 zile
de tranzactionare. Modelul incearca sa anticipeze daca perioada urmatoare va fi
calma sau agitata.

Pentru un student, o analogie simpla este prognoza meteo:

- pretul este ca temperatura exacta de maine;
- volatilitatea este ca nivelul de instabilitate al vremii;
- modelul nostru nu spune exact "pretul va fi X", ci "miscarea va fi mica sau mare".

### Anti-leakage (prevenire scurgere de informatii)

Toate transformarile sunt **fit-uite exclusiv pe setul de antrenare**:

- `StandardScaler` pentru features: fit pe `X_train`, `transform` pe `X_val` si `X_test`
- Normalizarea tintelor: media si std calculate pe `y_train`, aplicate pe toate split-urile
- Fereastra de intrare (`X`) se termina intotdeauna **inainte** de ziua tintei
- HPO optimizeaza pe validare, nu pe test

Un mod simplu de a intelege anti-leakage: modelul nu are voie sa invete folosind
informatii pe care nu le-ar fi avut in momentul real al predictiei. Daca am folosi
date din 2024 pentru a standardiza sau a alege hiperparametri inainte de antrenare,
am obtine rezultate prea optimiste.

---

## 6. Arhitecturi de modele

### LSTM (`nyse_vol/models/lstm.py`)

Model LSTM stivuit (stacked) cu cap de regresie:

```
Intrare: (batch, 60, 9)           # 60 zile x 9 features

LSTM stivuit (num_layers straturi, optional bidirectional)
  → Ultima stare ascunsa: (batch, hidden_size x [1 sau 2])

Cap dens: Linear(hidden_size) -> Activation -> Dropout -> Linear(4)
  → Iesire: (batch, 4)            # log-volatilitate pentru h=1,5,10,20
```

**Parametri default:** `hidden_size=64`, `num_layers=2`, `dropout=0.2`,
`bidirectional=False`, `head_activation="tanh"`.

### Cand este util LSTM?

LSTM este potrivit cand ordinea observatiilor conteaza. In acest proiect, ordinea
zilelor conteaza foarte mult: o zi volatila recenta poate fi mai relevanta decat
o zi volatila de acum doua luni. LSTM incearca sa pastreze in starea sa interna
informatia importanta din trecut.

### Encoder-Decoder cu Atentie (`nyse_vol/models/seq2seq_attention.py`)

Arhitectura Seq2Seq cu mecanism de atentie, generand fiecare orizont autoregresiv:

```
Intrare: (batch, 60, 9)

[ENCODER]
LSTM stivuit (optional bidirectional)
  --> Stari ascunse pentru toate cele 60 pozitii: (batch, 60, enc_dim)
  --> Ultima stare --> bridge linear --> stare initiala decoder

[DECODER — 4 pasi autoregresivi, unul per orizont]
La fiecare pas:
  1. Atentie peste cele 60 stari ale encoderului:
       Bahdanau (aditiv):      score = v^T * tanh(W1*s + W2*h_enc)
       Luong (multiplicativ):  score = s^T * W * h_enc
     --> ponderi de atentie pe fereastra de 60 zile (suma = 1)
  2. Vector de context = suma ponderata a starilor encoderului
  3. LSTMCell(predictia_anterioara + context) --> noua stare
  4. Linear(stare + context) --> predictia pentru orizontul curent

Iesire: (batch, 4)
```

**Avantaj fata de LSTM simplu:** mecanismul de atentie permite decoderului sa se
focalizeze pe zilele cele mai relevante din fereastra de 60 zile pentru fiecare
orizont. Heatmap-ul de atentie (generat la pasul 3) vizualizeaza aceste ponderi.

### Cum se citeste atentia?

Heatmap-ul de atentie arata ce zile din fereastra de 60 de zile au primit pondere
mai mare. Zonele mai intense inseamna ca modelul s-a uitat mai mult la acele zile.
Nu trebuie interpretat ca o explicatie perfecta a deciziei modelului, dar este util
pentru a vedea daca modelul acorda importanta unor perioade recente sau unor spike-uri
de volatilitate.

### Baseline GARCH

Model econometric clasic GARCH(p,q) pentru varianta conditionala:

```
h_{t+1} = omega + alpha_1*r_t^2 + ... + alpha_p*r_{t-p+1}^2
                + beta_1*h_t + ... + beta_q*h_{t-q+1}

vol_garch_h1(t) = sqrt(h_{t+1})
vol_garch_h5(t) = sqrt( mean(h_{t+1}, h_{t+2}, ..., h_{t+5}) )
```

**Evaluare walk-forward cu actualizare zilnica:** dupa fiecare refit (la 21 zile),
parametrii (omega, alpha, beta) sunt fixati iar h_t este actualizata ZILNIC cu
randamentul observat. Astfel predictiile variaza natural zi de zi (nu in trepte).

### Modelul Naiv (Persistence Baseline)

```
pred_naiv(t + h) = vol_garman_klass(t)    pentru orice h
```

Presupune ca volatilitatea de maine este identica cu cea de azi. Este surprinzator
de competitiv pe h=1 (autocorelatia volatilitatii la lag 1 este ridicata).
Orice model serios trebuie sa il depaseasca pentru a aduce valoare practica.

### De ce avem nevoie de baseline-uri?

Fara baseline-uri, nu stim daca un model neural este cu adevarat bun. Un LSTM poate
parea impresionant doar pentru ca este mai complex, dar intrebarea corecta este:

> Este mai bun decat o regula simpla?

De aceea sunt incluse GARCH si modelul Naiv. Ele ofera repere clare pentru comparatie.

---

## 7. Antrenare si HPO

### Bucla de antrenare (`nyse_vol/train/trainer.py`)

- **Loss:** MSE pe tintele standardizate (log-volatilitate normalizata)
- **Optimizatori disponibili:** Adam, AdamW, RMSprop, SGD
- **Early stopping:** patience = 6 epoci fara imbunatatire a `val_loss`
- **LR Scheduler:** `ReduceLROnPlateau` cu `factor=0.5`, `patience=2`
- **Gradient clipping:** norma maxima = 1.0 (stabilitate pentru RNN)
- **Cel mai bun checkpoint:** restaurat automat la sfarsitul antrenarii

### HPO — Random Search (`nyse_vol/train/hpo.py`)

```
Pentru fiecare din N trials:
  1. Alege aleator o configuratie din spatiul de cautare
  2. Antreneaza modelul pentru `--epochs` epoci (implicit 10)
  3. Evalueaza val_loss pe setul de VALIDARE
  4. Logeaza configuratia si rezultatele in CSV

Selecteaza configuratia cu cel mai mic val_loss.

Retreneaza cu configuratia optima pentru numarul complet de epoci (implicit 30).
Salveaza: lstm_best_hpo.pt / attention_best_hpo.pt + config JSON
```

**De ce HPO pe validare, nu pe test?** Testul este "secret" pana la evaluarea finala.
Alegerea hiperparametrilor pe test ar constitui trisat — modelul ar fi suprainvatat
pe test si nu ar generaliza pe date noi.

**Interpretarea `best_val`:** MSE pe tintele standardizate. Valoarea de referinta
este 1.0 (modelul care prezice intotdeauna media). `best_val < 1.0` inseamna ca
modelul a invatat ceva. Valori tipice bune: 0.3–0.6.

### Cum recunoastem overfitting-ul?

Overfitting-ul apare cand modelul invata prea bine datele de antrenare, dar nu mai
generalizeaza. In graficele de loss, un semn clasic este:

- `train_loss` continua sa scada;
- `val_loss` se opreste din scazut sau incepe sa creasca.

Early stopping opreste antrenarea cand validarea nu se mai imbunatateste. Astfel
pastram modelul care merge cel mai bine pe date nevazute in timpul antrenarii.

### Spatiu de cautare HPO

| Parametru             | Valori posibile           |
| --------------------- | ------------------------- |
| `optimizer`           | adam, adamw, rmsprop, sgd |
| `lr`                  | 3e-4, 1e-3, 3e-3          |
| `weight_decay`        | 0.0, 1e-5, 1e-4           |
| `hidden_size`         | 32, 64, 128               |
| `num_layers`          | 1, 2, 3                   |
| `dropout`             | 0.0, 0.2, 0.4             |
| `bidirectional`       | False, True               |
| `head_activation`     | tanh, relu                |
| `attention` (Seq2Seq) | bahdanau, luong           |

---

## 8. Evaluare si metrici

Toate metricile sunt calculate pe setul de test (> 2021-12-31), in **scala
volatilitatii brute** (dupa `exp()` aplicat pe tintele log).

| Metrica     | Formula                           | Interpretare                                                                                |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------- | --- | ------------------------------------------------------------------- |
| **RMSE**    | `sqrt(mean((y - y_pred)^2))`      | Eroarea medie patratica, aceleasi unitati ca volatilitatea. Mai mic = mai bun.              |
| **MAE**     | `mean(                            | y - y_pred                                                                                  | )`  | Eroarea medie absoluta. Mai robusta la outlieri. Mai mic = mai bun. |
| **R²**      | `1 - SS_res / SS_tot`             | Proportia variatiei explicate. 1.0 = perfect, 0.0 = ca media, < 0 = mai rau.                |
| **QLIKE**   | `mean(sigma^2 / h^2 + log(h^2))`  | Penalizeaza asimetric subestimarea (mai periculoasa in risk management). Mai mic = mai bun. |
| **Dir.Acc** | `mean(sign(dy) == sign(dy_pred))` | % din zile cu directia schimbarii corecta. 0.5 = aleator, > 0.5 = informativ.               |

**De ce QLIKE?** In managementul riscului, subestimarea volatilitatii (a crede ca
piata e calma cand nu e) este mult mai periculoasa decat supraestimarea. QLIKE
penalizeaza asimetric acest scenariu, spre deosebire de RMSE si MAE care sunt simetrice.

### Cum alegem "cel mai bun" model?

Nu exista o singura metrica perfecta. In practica:

- pentru eroarea generala, ne uitam la **RMSE** si **MAE**;
- pentru risc financiar, ne uitam la **QLIKE**;
- pentru explicarea variatiei, ne uitam la **R²**;
- pentru directia miscarii, ne uitam la **Dir.Acc**.

Daca un model are RMSE mic, dar QLIKE mare, inseamna ca poate face erori periculoase
in estimarea riscului. Daca are R² negativ, inseamna ca nu explica bine seria si poate
fi mai slab decat o predictie foarte simpla.

O concluzie solida se bazeaza pe mai multe metrici si pe grafice, nu doar pe o singura
valoare din tabel.

---

## 9. Grafice generate

### `artifacts/plots/antrenare/`

- `lstm_loss.png`, `attention_loss.png` — curbe de loss train/validare (modele referinta)
- `lstm_best_hpo_loss.png`, `attention_best_hpo_loss.png` — curbe de loss dupa HPO
- `attention_heatmap.png` — heatmap ponderi de atentie (60 zile x 4 orizonturi)

### `artifacts/plots/comparatii/`

- `compare_rmse.png` — bar chart RMSE per model si orizont (h=1,5,10,20); cel mai bun cu contur portocaliu
- `compare_qlike.png` — bar chart QLIKE per model si orizont

### `artifacts/plots/predictii/`

- `pred_vs_true_{model}_{symbol}.png` — volatilitate reala vs. prezisa pentru
  simbolul cel mai frecvent din test, cu trend 21-zile suprapus

### `artifacts/plots/stocuri/` (generate de pasul 6)

- `stock_{SYM}_h1_toate_modelele.png` — subplot separat per model (Real vs. Prezis
  la h=1), fiecare cu RMSE in titlu; mai usor de comparat decat linii suprapuse
- `stock_{SYM}_toate_orizonturile.png` — 4 subploturi (h=1,5,10,20) pentru cel
  mai bun model dupa RMSE pe tot setul de test
- `stock_{SYM}_metrici.png` — bar charts RMSE/MAE/R²/QLIKE per model la h=1;
  cel mai bun per metrica evidentiat cu contur portocaliu

### Ce grafice sunt cele mai utile in raport?

Pentru prezentarea proiectului, cele mai relevante sunt:

1. Curbele de loss, pentru a arata cum a invatat modelul.
2. Comparatia RMSE/QLIKE, pentru a vedea diferentele intre modele.
3. Predictie vs. real, pentru a vedea vizual daca modelul urmareste seria.
4. Graficele per stock, pentru exemple concrete pe actiuni alese.

---

## 10. Cum se ruleaza

### Cerinte preliminare

```bash
pip install torch numpy pandas scikit-learn matplotlib arch statsmodels
```

Datele brute NYSE trebuie sa fie in `data_raw/NYSE_AAAA.csv` pentru fiecare an.

Pentru structura implementata in loader, datele pot fi puse si in directoare de tip
`data_raw/NYSE_2001/NYSE_20010102.csv` sau in arhive `data_raw/NYSE_2001.zip`.
Aceasta este varianta recomandata pentru fisierele zilnice.

### Rulare completa (recomandata)

```bash
# Pas 1: preprocesare si feature engineering
python scripts/01_prepare_data.py

# Optional: exporta CSV-uri inspectabile (features_all.csv, panel_silver.csv)
python scripts/01_prepare_data.py --export

# Pas 2: baseline GARCH (testeaza 8 combinatii, alege cel mai bun)
python scripts/02_train_garch.py

# Pas 3: modele NN cu hiperparametri impliciti (modele de referinta)
python scripts/03_train_nn.py --epochs 30

# Pas 4: HPO - gaseste configuratia optima, reantrenare completa
python scripts/04_run_hpo.py --model lstm --trials 10 --epochs 15
python scripts/04_run_hpo.py --model attention --trials 10 --epochs 15

# Pas 5: evaluare si comparare toate modelele pe setul de test
python scripts/05_evaluate_compare.py

# Pas 6: grafice detaliate per stock
python scripts/06_select_and_plot.py --symbols JPM GS BAC
# SAU selectie interactiva:
python scripts/06_select_and_plot.py
```

### Rulare rapida (fara HPO)

```bash
python scripts/01_prepare_data.py
python scripts/02_train_garch.py
python scripts/03_train_nn.py --epochs 20 --model lstm
python scripts/05_evaluate_compare.py
python scripts/06_select_and_plot.py --symbols JPM GS
```

Aceasta varianta este utila pentru verificare initiala. Ruleaza mai repede, dar
rezultatele pot fi mai slabe decat cele obtinute dupa HPO, deoarece foloseste
hiperparametrii impliciti.

### Rulare pe un subset de simboluri

```bash
export SYMBOLS="JPM,GS,BAC,C"
python scripts/01_prepare_data.py
python scripts/03_train_nn.py
```

Subsetul de simboluri este util cand vrem sa testam codul rapid sau cand lucram pe
un calculator fara GPU. Pentru rezultate finale, este mai bine sa folosim mai multe
simboluri, deoarece modelul vede mai multe exemple.

### Rulare pe GPU (CUDA)

```bash
export DEVICE=cuda
python scripts/03_train_nn.py
python scripts/04_run_hpo.py --model lstm --trials 20 --epochs 20
```

### Rulare pe Kaggle (GPU T4 x2)

```python
import os
os.environ["DATA_DIR"]      = "/kaggle/input/nyse-data"
os.environ["ARTIFACTS_DIR"] = "/kaggle/working/artifacts"
os.environ["DEVICE"]        = "cuda"
```

```bash
!python scripts/01_prepare_data.py --export
!python scripts/02_train_garch.py
!python scripts/03_train_nn.py --epochs 50
!python scripts/04_run_hpo.py --model lstm      --trials 15 --epochs 20 --retrain-epochs 50
!python scripts/04_run_hpo.py --model attention --trials 15 --epochs 20 --retrain-epochs 50
!python scripts/05_evaluate_compare.py
!python scripts/06_select_and_plot.py --symbols JPM GS BAC C AXP
```

Pe Kaggle, directoarele de input sunt read-only, iar `ARTIFACTS_DIR` trebuie setat
catre `/kaggle/working/artifacts`, pentru ca modelele si graficele sa poata fi
salvate.

---

## 11. Configurare

Toate setarile centrale sunt in `nyse_vol/config.py`:

```python
HORIZONS         = [1, 5, 10, 20]    # orizonturi de predictie (zile bursiere)
WINDOW           = 60                 # lungimea ferestrei de intrare (zile)
TARGET_ESTIMATOR = "garman_klass"     # estimatorul folosit ca tinta
TRAIN_END        = "2018-12-31"       # limita train
VAL_END          = "2021-12-31"       # limita validare; testul = dupa aceasta data
SEED             = 42
DEVICE           = "cpu"              # suprascribil prin env var DEVICE
```

### Variabile de mediu

| Variabila       | Default      | Descriere                           |
| --------------- | ------------ | ----------------------------------- |
| `DATA_DIR`      | `data_raw/`  | Directorul cu fisierele NYSE CSV    |
| `ARTIFACTS_DIR` | `artifacts/` | Directorul cu artefacte generate    |
| `SYMBOLS`       | toate 30     | Lista de simboluri, virgula-separat |
| `DEVICE`        | `cpu`        | `cpu` sau `cuda`                    |

---

## 12. Cerinte

```
torch >= 2.0
numpy
pandas
scikit-learn
matplotlib
arch          # pentru modele GARCH
statsmodels   # pentru testele de stationaritate (ADF, KPSS)
```

Instalare:

```bash
pip install torch numpy pandas scikit-learn matplotlib arch statsmodels
```

---

## Note tehnice

### De ce fereastra de 60 de zile?

60 de zile (~3 luni) captureaza:

- Autocorelatia volatilitatii (efectul GARCH se propaga pe zeci de zile)
- Sezonalitate intratrimestriala (efecte de raportare financiara trimestriala)
- Regimuri de volatilitate care dureaza saptamani–luni

### De ce Garman-Klass ca tinta si nu close-to-close?

Garman-Klass utilizeaza toate cele 4 preturi OHLC si are o **eficienta de 5-8x**
mai mare decat close-to-close (acelasi nivel de precizie cu de 5 ori mai putine date).
Close-to-close ignora toata miscarea intraday, subevaluand semnificativ volatilitatea
in zilele cu miscari mari (ex: crize, earnings surprises).

### Comportamentul modelului Naiv

Modelul Naiv prezice `vol(t+h) = vol_garman_klass(t)` pentru orice h. In grafice
va arata ca o linie aproape identica cu realitatea, decalata cu exact 1 zi — acesta
este comportamentul corect (nu o eroare). Volatilitatea are autocorelare ridicata
la lag 1. Un RMSE al oricarui model mai mare decat Naivul inseamna ca modelul nu
aduce valoare practica.

### Lag-ul vizibil la modelele LSTM

Modelele LSTM si Attention pot prezenta un mic decalaj vizibil fata de seria reala
— aceasta este o caracteristica a modelelor secventiale care netezesc semnalul.
Este in mare parte inevitabil; ceea ce conteaza pentru evaluare este RMSE global,
nu sincronizarea exacta cu spikele extreme individuale.

---

### Ce trebuie retinut

Proiectul are o idee centrala simpla:

> Din ultimele 60 de zile de comportament bursier, incercam sa estimam volatilitatea
> urmatoarelor 1, 5, 10 si 20 de zile.

Modelele neuronale sunt folosite pentru ca pot invata dependente temporale, iar
baseline-urile sunt folosite pentru a verifica daca modelele complexe chiar aduc
un avantaj.

### Posibile imbunatatiri

Cateva directii naturale de continuare:

- testarea altor orizonturi, precum 30 sau 60 de zile;
- folosirea unui numar mai mare de simboluri;
- adaugarea unor indicatori tehnici, precum medii mobile sau RSI;
- compararea cu alte arhitecturi, de exemplu GRU sau Transformer;
- antrenarea separata pe sectoare, cum ar fi banci, energie sau tehnologie;
- analiza erorilor in perioade de criza, cand volatilitatea creste brusc.
