# AI prepoznavanje znakov – Kamen, škarje, papir

## Opis projekta

Projekt omogoča prepoznavanje treh znakov z uporabo spletne kamere in umetne inteligence:

- ✊ Kamen
- ✌️ Škarje
- ✋ Papir

Program uporablja knjižnico **MediaPipe** za zaznavanje 21 ključnih točk na roki. Njihove koordinate obdela lastna nevronska mreža, izdelana in naučena s knjižnico TensorFlow.

Model V3 je bil naučen na 900 vzorcih in je pri dodatnem testiranju na 150 novih vzorcih dosegel **91,33 % natančnost**.

## Navodila za namestitev

### 1. Prenos projekta

Na GitHubu kliknite **Code → Download ZIP**.

Preneseno datoteko razširite v izbrano mapo.

### 2. Odpiranje ukaznega poziva

Odprite mapo projekta v Raziskovalcu, kliknite naslovno vrstico, vpišite `cmd` in pritisnite Enter.

### 3. Ustvarjanje virtualnega okolja

```cmd
python -m venv .venv
```

Aktivirajte virtualno okolje:

```cmd
.venv\Scripts\activate.bat
```

### 4. Namestitev knjižnic

```cmd
python -m pip install -r requirements.txt
```

### 5. Zagon programa

```cmd
python prepoznavanje_v3.py
```

Odprlo se bo okno spletne kamere.

Pred kamero pokažite enega izmed treh znakov. Program bo prikazal prepoznani znak in ocenjeno zanesljivost napovedi.

Za izhod pritisnite tipko **Q**.

## Datoteke projekta

| Datoteka               | Opis                               |
| ---------------------- | ---------------------------------- |
| `model_v3.keras`       | Naučena nevronska mreža            |
| `hand_landmarker.task` | Model MediaPipe za zaznavanje roke |
| `razredi_v3.json`      | Seznam znakov                      |
| `prepoznavanje_v3.py`  | Program za prepoznavanje           |
| `requirements.txt`     | Potrebne knjižnice                 |

## Rezultati testiranja

| Znak       | Pravilne napovedi |  Natančnost |
| ---------- | ----------------: | ----------: |
| Kamen      |             44/50 |        88 % |
| Škarje     |             46/50 |        92 % |
| Papir      |             47/50 |        94 % |
| **Skupaj** |       **137/150** | **91,33 %** |

Rezultati predstavljajo uspešnost na dodatnih testnih vzorcih in ne zagotavljajo enake natančnosti v vseh okoliščinah.

## Opomba

Za uporabo programa ni potrebno ponovno učenje nevronske mreže, saj je naučeni model že vključen v projekt.
