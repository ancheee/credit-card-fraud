# Credit Card Fraud Detection — Duboko Ucenje

Projekat (seminarski rad) za predmet **Duboko ucenje i neuronske mreze**.

## O radu

Zadatak je bio primeniti neuronske mreze na problem detekcije prevara u finansijskim
transakcijama. Glavna teskoca kod ovog dataseta nije sama arhitektura mreze, vec cinjenica
da je dataset izrazito nebalansiran — svega 492 od 284,807 transakcija (0.17%) su prevare.
Zbog toga je vecina rada posvecena razlicitim nacinima da se ta nebalansiranost tretira, i
poredjenju koliko svaki od njih zapravo pomaze.

U notebook-u sam isprobala sedam pristupa, redom od najjednostavnijeg ka
slozenijem: dummy baseline, logisticku regresiju, obican MLP (bez ikakvog tretmana
nebalansiranosti — da se vidi zasto je to problem), MLP sa class weightingom, podesavanje
praga odlucivanja (threshold tuning), MLP treniran na SMOTE-ovanom skupu, i na kraju
autoencoder kao unsupervised pristup istom problemu.

## Struktura projekta

```
credit_card_fraud/
├── fraud_detection.ipynb   ← glavni notebook (generisan skriptom ispod)
├── create_notebook.py      ← skripta koja generise notebook
├── requirements.txt        ← potrebne biblioteke
├── README.md               ← ovaj fajl
└── creditcard.csv          ← dataset (preuzeti rucno, vidi ispod)
```

Notebook nije pisan celija po celija u Jupyter-u, vec je generisan pomocu
`create_notebook.py` — to mi je bilo lakse za verzionisanje i za naknadne izmene teksta
u markdown celijama, pa fajl `create_notebook.py` sadrzi identican sadrzaj kao i notebook.

## Kako preuzeti dataset

Dataset nije ukljucen u ovaj repozitorijum jer je prevelik (~150 MB).

1. Otici na: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Preuzeti `creditcard.csv`
3. Staviti ga u isti folder kao notebook

## Pokretanje

```bash
pip install -r requirements.txt

# Notebook je vec generisan, ali ako se menja create_notebook.py,
# potrebno je ponovo ga pokrenuti:
python create_notebook.py

# Pokretanje Jupyter-a:
jupyter lab fraud_detection.ipynb
```

Treniranje svih modela (narocito neuronskih mreza) traje neko vreme na CPU-u — u pitanju
je nekoliko minuta po modelu, u zavisnosti od masine.

## Implementirani modeli

| # | Model | Nacin tretmana nebalansiranosti |
|---|---|---|
| 1 | Dummy Baseline | — (uvek predvidja vecinsku klasu, sluzi kao donja granica) |
| 2 | Logisticka regresija | Class weighting |
| 3 | Basic MLP | Bez tretmana (namerno, radi poredjenja) |
| 4 | Weighted MLP | Class weighting |
| 5 | Weighted MLP + threshold tuning | Class weighting + podesen prag |
| 6 | SMOTE + MLP | Sintetisko uvecanje manjinske klase |
| 7 | Autoencoder | Unsupervised (uci samo od legitimnih transakcija) |

## Metrike koje sam koristila

Accuracy nije relevantna metrika za ovaj problem (objasnjeno detaljnije u notebook-u), pa
su umesto nje koriscene:

- **Recall** — koliko od stvarnih prevara je model prepoznao (najbitnija metrika ovde,
  jer je propustena prevara skuplja od laznog alarma)
- **Precision** — koliko su predikcije "prevara" zaista tacne
- **F1-Score** — balans izmedju Precision i Recall
- **PR-AUC** — opsta metrika koja uzima u obzir sve moguce pragove, korisna kod
  nebalansiranih podataka
- **ROC-AUC** — dodatna metrika, prikazana radi potpunosti

## Glavni zakljucci (detaljnije u poslednjoj celiji notebook-a)

- Accuracy je ovde varljiva — dummy model ima 99.83% accuracy i 0% recall
- Class weighting jasno poboljsava recall u odnosu na neweighted MLP
- Threshold tuning na validacionom skupu dodatno pomera odnos precision/recall u zeljenom
  pravcu, u zavisnosti od toga sta se optimizuje
- SMOTE je alternativan nacin balansiranja, poredjen direktno sa class weightingom
- Autoencoder ne koristi labele tokom treniranja, a ipak uspeva da razlikuje prevare na
  osnovu reconstruction error-a — korisno kada labele ne bi bile dostupne

## Ogranicenja

Hiperparametri nisu sistematski tunirani (nema grid/random search-a), vec su postavljeni
na uobicajene vrednosti iz literature. Rezultati takodje zavise od jednog fiksnog
train/val/test splita (SEED=42) — za pouzdaniju procenu bi trebalo uraditi
cross-validation, sto nije bio fokus ovog rada.

## Biblioteke

- Python 3.10+
- TensorFlow 2.13+
- scikit-learn 1.3+
- imbalanced-learn 0.11+
- pandas, numpy, matplotlib, seaborn
# credit-card-fraud
