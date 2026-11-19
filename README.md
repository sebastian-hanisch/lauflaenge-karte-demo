# Karte und Lauflänge: was „ARL = 370“ verschweigt – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lauflaenge-karte-demo.streamlit.app/)**

Erste Seite der Reihe **Änderungserkennung: wie schnell merkt man, dass sich etwas geändert hat?** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Konzepte-Reihe beginnt hier, es gibt keinen Vorgänger.
Das Vehikel ist **ausgedacht**: unabhängige Normalwerte, Mittel und Streuung bekannt, ein Sprung des Mittelwerts an einer bekannten Stelle. Es gibt keine reale Datenquelle; in den gelesenen Quellen gab es keine Messung, ob eine solche Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert.
Gezählt werden Beobachtungen bis zum Alarm; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. Verwandt, aber nicht nachgebaut: `niveausprung-demo` (CUSUM auf Prognoseresiduen mit Kalender und Bestandskosten; dort „Meldungen je Jahr“, hier „Alarme je Fenster“).

## Warum dieses Problem

Eine Kontrollkarte mit Grenzen ±3σ meldet in diesem Modell (unabhängig, normal, Ziel und Streuung bekannt, zweiseitig) im Mittel erst nach 370,4 Beobachtungen einen Fehlalarm („ARL₀“, zero-state). Aber die Wartezeit ist **geometrisch verteilt**: der Median liegt bei 257 Beobachtungen, 23,69 % der Reihen haben schon innerhalb der ersten 100 Beobachtungen einen Alarm, 6,70 % bleiben über 1000 Beobachtungen ruhig, und in einem Fenster von 1000 Beobachtungen gibt es im Mittel 2,70 Alarme, in 13,68 % der Fenster fünf oder mehr.
Ein Sprung um 1σ braucht im Mittel 43,9 Beobachtungen (zero-state; bei dieser gedächtnislosen Karte zugleich Pollak und Lorden; Median 31, 95-%-Quantil 130), ein Sprung um 0,5σ bleibt in 52,40 % der Reihen länger als 100 Beobachtungen unentdeckt.
Zusatzregeln („nach Art der Western-Electric-Regeln“, Fassung unten) senken die ARL₀ auf 91,75 und sind bei **gleicher** ARL₀ gegen die einfache Karte bei δ = 1 deutlich schneller (9,22 gegen 16,32 Beobachtungen, zero-state), verlieren aber ab δ\* = 2,256 (δ = 3: 1,67 gegen 1,48).

**Kernfragen der App:** (1) Wann schlägt die Karte an? (2) Was heißt „ARL = 370“ für eine einzelne Reihe? (3) Wie viele Fehlalarme gibt es in einem Fenster? (4) Wie lange dauert es nach einem Sprung? (5) Was kosten Zusatzregeln?
Gilt für zweiseitige Karten mit bekanntem Ziel und bekannter Streuung, unabhängige Normalwerte; Zähleinheiten sind **Beobachtungen** (Lauflänge, Verzögerung), **Reihen** (Anteile), **Läufe** (Simulation) und **Fenster** zu 1000 Beobachtungen (Alarmzahl, Neustart nach Alarm ohne Vorgeschichte).

## Befunde und Korrekturen gegenüber Plan und Vormessung

Die Kernzahlen des Plans (Abschnitte A bis F unten) sind im Bau neu gerechnet worden (Ketten, geschlossene Formen, Aufzählung) und stimmen bis auf die Darstellung überein; die Simulationszahlen der Vormessung (andere Zufalls-Seeds) und der Ausblick auf den CUSUM sind nicht neu gerechnet. Abweichungen und neue Befunde:

* **„Etwa ein Drittel der frühen Fehlalarme“ war zu stark.** Bei gleicher ARL₀ = 91,75 alarmieren die vier Regeln innerhalb der ersten 10 Beobachtungen in 7,47 % der Reihen, die einfache Karte in 10,38 %: das sind 28 % weniger (nicht 33 %).
* **Zustandszahlen der Kette:** der Plan nennt 2041 / 1545 / 1497 / 1497 Zustände (Laufzähler immer mitgeführt). Diese Demo führt nur die gebrauchte Vorgeschichte mit (zwei Zonen für Regel 2, vier für Regel 3, den Laufzähler nur mit Regel 4) und kommt auf 1 / 41 / 1221 / 1497 Zustände für die Regelsätze 1 / 1+2 / 1+2+3 / 1+2+3+4 mit denselben ARL-Werten; mit Regel 4 = 9 und 10 in Folge sind es 1589 und 1681.
  Die Fensterkette (Testhilfe, ganz anderer Aufbau) hat 1555 / 1233 / 1221 / 15 165 Zustände.
* **Maße der Verzögerung für alle drei Zusatzsätze:** der Plan hat „zero-state = Pollak = Lorden“ nur für die vier Regeln gemessen; der Test deckt jetzt die Regelsätze 1+2, 1+2+3 und 1+2+3+4 bei δ = 0,5; 1; 2; 3 ab (12 Zellen, Abstand höchstens 1e-9), in allen ist steady-state kleiner. Das gilt für diese Fassung der Regeln und dieses Raster, nicht allgemein.
* **Das Näherungsmaß W/ARL₀ für die Alarmzahl** weicht bei Fenstern zu 1000 Beobachtungen höchstens um 0,5 % von der Erneuerungsrechnung ab (vier Regeln: 10,899 gegen 10,866, also 0,3 %); bei Fenstern zu 200 Beobachtungen sind es bis rund 1,5 % (vier Regeln: 2,180 gegen 2,147).
* **Regler:** Der Plan nannte „Sprung nach ν Beobachtungen (0 = keiner)“. Hier heißt ν = 0 „die Änderung liegt vor der ersten Beobachtung“ (so die Konvention der Konzepte-Reihe), und „kein Sprung“ ist δ = 0. Ein Regler „Ziel-ARL₀“ (20 bis 5000) gibt es nicht: die Schwelle c (2,0 bis 4,0 σ) hat einen Knopf für ARL₀ = 370 und die Tabelle „ARL₀ → Anteil mit Alarm innerhalb der ersten t₀“ deckt die Abhängigkeit ab. Die Simulation läuft auf Klick (Knopf mit Zeitschätzung), die Anzeige ist die Verteilungsfunktion mit den simulierten Punkten statt eines Histogramms.
* **Simulationen sind Bänder:** zwei andere Zufalls-Seeds als in der Vormessung geben bei 3000 Läufen der einfachen Karte (Messung dieser README, nicht im Test): Zufalls-Seed 41001: Mittel 366,5 ± 6,8 gegen exakt 370,4, 23,70 % der Reihen mit Alarm innerhalb von 100 gegen exakt 23,69 %; Zufalls-Seed 41002: 382,5 ± 6,9 und 21,87 % (z = −2,35 gegen exakt).
  Fenster zu 1000 Beobachtungen, je 1000 Fenster: Anteil ohne Alarm 6,20 % und 8,00 % gegen exakt 6,70 %, Mittel 2,739 ± 0,053 und 2,610 ± 0,050 gegen exakt 2,700. Der Standardfehler eines Anteils aus 3000 Läufen oder 1000 Fenstern liegt bei rund 0,8 Prozentpunkten (bei p = 23,7 % bzw. 6,7 %; bis 1,6 bei 1000 Fenstern und p = 50 %), ± 2 Standardfehler sind rund 1,5 Prozentpunkte.
* **Fassung von Regel 4 = 10 in Folge:** die Laufzähler-Kette ist gerechnet, die Fensterkette dazu braucht rund 100 s und läuft deshalb im Test nur für 8 und 9 in Folge.

## Modell

Unabhängige Beobachtungen $x_t \sim N(0, 1)$ für $t \le \nu$ und $N(\delta, 1)$ danach, Ziel 0 und Streuung 1 bekannt. Alarmregeln (Fassung dieser Demo, **nach Art der Western-Electric-Regeln**, keine Wiedergabe einer gelesenen Quelle):
Regel 1: $|x| > 3$ (die einfache Karte: Schwelle $c$, wahlweise einseitig). Regel 2: zwei von drei aufeinanderfolgenden Beobachtungen jenseits 2σ auf derselben Seite. Regel 3: vier von fünf jenseits 1σ auf derselben Seite. Regel 4: 8 (oder 9, 10) in Folge auf derselben Seite der Mittellinie. Die Regelsätze 1, 1+2, 1+2+3, 1+2+3+4 sind geschachtelt.
**Lauflänge** $T$ = Beobachtungen bis einschließlich der alarmierenden; **ARL₀** = $E[T]$ ohne Änderung, **ARL₁** mit Sprung; **Fehlalarm** = ein Alarm vor der Änderung, je Zeit gezählt (nicht Anteil falscher unter den positiven, nicht „mindestens ein Fehlalarm unter m Tests“). ν = Zahl der Beobachtungen vor der Änderung, Verzögerung $D = T - \nu \ge 1$.
Maße: **zero-state** (Start ohne Vorgeschichte), **steady-state** (quasistationär: Grenzwert von $E[D]$ für $\nu \to \infty$, bedingt auf „kein Alarm davor“; nie das Shiryaev-Maß, das zyklisch ist und hier nicht gerechnet wird), **Pollak** (Maximum über ν) und **Lorden** (schlechtester erreichbarer Zustand). Das Maß steht in jedem Satz der App. Die Formeln mit Lesehilfe stehen im 📐-Expander der App.
„Karte“ allein heißt in `niveausprung-demo` die Heatmap Rauschen × Sprung; hier steht deshalb „Kontrollkarte“. „Lauflänge“ ist hier immer die Zahl der Beobachtungen bis zum Alarm; ein einzelner Simulationslauf heißt „Lauf“.

## Methodik

* **Einfache Karte:** exakt. Mit $p(\delta) = \Phi(\delta - c) + \Phi(-c - \delta)$ ist die Lauflänge geometrisch: $P(T > t) = (1-p)^t$, ARL $= 1/p$. Die Karte als Kette mit **einem** Zustand gibt dasselbe.
* **Zusatzregeln:** exakt als absorbierende Markov-Kette (`lfk_chain.py`). Zustand = Zonen der letzten Beobachtungen (Zonengrenzen 0, ±1, ±2, ±3 sind die Regelgrenzen, deshalb kein Diskretisierungsfehler) und der Lauf auf einer Seite. Überlebensfunktion $e_0^\top R^t \mathbf{1}$, ARL aus $(I - R)^{-1}\mathbf{1}$ (`scipy.sparse`).
* **Alarme je Fenster:** Neustart nach Alarm ohne Vorgeschichte (Festlegung dieser Seite: ARL ist nur sinnvoll, wenn bei Alarm angehalten wird, Zwetsloot u. a. 2024); Erneuerungsrechnung $P(N_W \ge k) = P(S_k \le W)$ aus der exakten Lauflängenverteilung, bei der einfachen Karte zusätzlich Binomial als zweiter Weg.
* **Orakel (anderer Rechenweg):** geschlossene Form ↔ Kette mit einem Zustand ↔ exakte Brüche; Laufzähler-Kette ↔ Fensterkette (ohne Laufzähler) in allen vier Regelsätzen bei δ = 0; 1; 3 (Regel 4 mit 9 in Folge: δ = 0); Aufzählung aller 8^6 Zonenfolgen live und aller 8^8 eingefroren (`lfk_enum.json`, `tools/lfk_make_enum.py --check`); Regeln an Rohwerten ↔ Schleifen; Erneuerung ↔ Binomial; Simulation im Band.
* **Zufall:** SplitMix64 (`lfk_rng.py`, bitgleich auf jeder Plattform). Die Einzelreihe nimmt `normal()` (Box-Muller, etwa 0,1 s bei 1000 Beobachtungen), die Simulation von Lauflängen und Fenstern `random()` mit `NormalDist().inv_cdf` (rund 1 µs je Normalzahl); Ergebnisse dieser Simulationen sind Bänder, nie exakte Vergleiche. Für die einfache Karte wird die Zufallszahl direkt mit den Randschwellen verglichen (gleichwertig zu $|x| > c$), die Zusatzregeln arbeiten an den Rohwerten.

## Befunde (gemessen, keine Behauptungen)

Zweiseitige Karten, unabhängige Normalwerte mit bekanntem Ziel, wenn nicht anders genannt zero-state; Zähleinheit Beobachtungen, Reihen (Anteile) oder Fenster zu 1000 Beobachtungen. Alle Zahlen exakt (geschlossene Form oder Kette), sofern nicht „Simulation“ dasteht; der Test steht in der Tabelle „Zahl → Test“ weiter unten.

**A. Einfache Karte, ±3σ: geometrisch verteilte Lauflänge.** ARL₀ = 370,398 (ARL₀ = 1/p mit p = 2Φ(−3)); einseitig 740,80. Schwelle für das **gesetzte** Ziel ARL₀ = 370: c = 2,9997 zweiseitig, c = 2,7818 einseitig (die ±3σ-Karte hat 370,398: zwei verschiedene Ziele).
Quantile (5 % / Median / 95 % / 99 %): 19 / 257 / 1109 / 1704 bei ±3σ, 19 / 257 / 1107 / 1702 bei ARL₀ = 370 gesetzt. P(T ≤ 100) = 23,69 % (±3σ) bzw. 23,71 % (370 gesetzt); „vor Beobachtung 100“ im engen Sinn, P(T ≤ 99), ist 23,48 % (±3σ) bzw. 23,50 % (370 gesetzt); P(T > 1000) = 6,70 %.

| ARL₀ (einfache Karte) | 50 | 100 | 200 | 370 | 500 | 1000 | 5000 |
|---|---|---|---|---|---|---|---|
| Anteil der Reihen mit Alarm innerhalb der ersten 100 Beobachtungen | 86,74 % | 63,40 % | 39,42 % | 23,71 % | 18,14 % | 9,52 % | 1,98 % |
| Median der Lauflänge | 35 | 69 | 139 | 257 | 347 | 693 | 3466 |

Zweiseitig mit Schwelle c: c = 2,0 gibt ARL₀ = 21,98 (99,05 % der Reihen mit Alarm innerhalb der ersten 100); c = 2,5: 80,52 (71,34 %); c = 3,0: 370,40 (23,69 %); c = 3,5: 2149,34 (4,55 %).

**B. Zusatzregeln: ARL₀ und Lauflängenverteilung exakt (zero-state).**

| Regeln | ARL₀ | Median | 5 % | 95 % | 99 % | P(T ≤ 10) | P(T ≤ 100) | P(T > 1000) | größter Abstand zur geometrischen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 370,398 | 257 | 19 | 1109 | 1704 | 2,67 % | 23,69 % | 6,70 % | 0,0000 |
| 1+2 | 225,438 | 157 | 13 | 673 | 1034 | 4,12 % | 35,80 % | 1,16 % | 0,0025 |
| 1+2+3 | 132,893 | 93 | 9 | 394 | 604 | 6,13 % | 52,86 % | 0,048 % | 0,0134 |
| 1+2+3+4 | 91,751 | 65 | 8 | 268 | 409 | 7,47 % | 66,68 % | 0,001 % | 0,0344 |

„Größter Abstand“ = größter Betrag der Differenz der Überlebensfunktion zur geometrischen mit gleichem Mittel. Die Lauflänge der Zusatzregeln ist also **nahezu**, aber nicht exakt geometrisch. Bei **gleicher** ARL₀ = 91,75 gleicht sie der der einfachen Karte (Schwelle c = 2,5459) nur grob: P(T ≤ 100) 66,68 % gegen 66,58 %, Median 65 gegen 64, 95-%-Quantil 268 gegen 274,
aber der größte Abstand der Verteilungsfunktionen ist 0,0344 bei t = 7; innerhalb der ersten 10 Beobachtungen alarmieren die Regeln in 7,47 % der Reihen gegen 10,38 %, innerhalb der ersten 20 in 17,39 % gegen 19,68 %. Mehr Regeln lösen das Problem „ARL ist ein Mittel“ nicht (gemessen an den drei Zusatzsätzen dieser Fassung); die Verteilung verschiebt sich wesentlich nur mit der ARL₀ (oben: 63,40 / 23,71 / 9,52 % bei ARL₀ = 100 / 370 / 1000).

**C. Fairer Vergleich bei gleicher ARL₀ und Kreuzung (zweiseitig, zero-state, ARL₁ in Beobachtungen, Verhältnis Regeln/einfach in Klammern).** Einfache Karte mit der ARL₀ des Regelsatzes: c = 2,8454 / 2,6727 / 2,5459 für 1+2 / 1+2+3 / 1+2+3+4.

| δ | 1+2: Regeln / einfach | 1+2+3: Regeln / einfach | 1+2+3+4: Regeln / einfach |
|---|---|---|---|
| 0,5 | 77,72 / 100,86 (0,771) | 38,64 / 63,87 (0,605) | 27,33 / 46,42 (0,589) |
| 1 | 20,01 / 30,72 (0,651) | 10,95 / 21,13 (0,518) | 9,22 / 16,32 (0,565) |
| 2 | 3,646 / 5,026 (0,725) | 3,140 / 3,991 (0,787) | 3,128 / 3,418 (0,915) |
| 3 | 1,676 / 1,781 (0,941) | 1,667 / 1,592 (1,047) | 1,667 / 1,481 (1,125) |
| 4 | 1,166 / 1,142 (1,021) | 1,166 / 1,102 (1,058) | 1,166 / 1,079 (1,081) |

**Kreuzung** (ARL₁ gleich) bei δ\* = 3,455 / 2,705 / 2,256 (1+2 / 1+2+3 / 1+2+3+4); darüber ist die einfache Karte schneller. Verteilung der Verzögerung bei δ = 1 (vier Regeln gegen einfache Karte, gleiche ARL₀ = 91,75): Mittel 9,22 gegen 16,32, Median 8 gegen 11, 95 % innerhalb von 22 gegen 48, 99 % innerhalb von 32 gegen 73, P(D ≤ 10) 69,79 % gegen 46,85 %, **P(D > 50) 0,05 % gegen 4,24 %**.
Einfache Karte ±3σ (zweiseitig, geschlossen), ARL₁ / Median / 95 % / Anteil der Reihen mit Verzögerung über 100 Beobachtungen: δ = 0,5: 155,22 / 108 / 464 / **52,40 %**; δ = 1: 43,89 / 31 / 130 / 9,98 %; δ = 1,5: 14,97 / 11 / 44 / 0,10 %; δ = 2: 6,30 / 5 / 18 / unter 0,001 %; δ = 3: 2,00 / 1 / 5 / unter 0,001 %.

**D. Maße der Verzögerung bei den Zusatzregeln (vier Regeln, ARL₀ = 91,75, ν-Konvention der Konzepte-Reihe).** zero-state, Pollak (Maximum über ν ≤ 600, erreicht bei ν = 0) und Lorden (Maximum über alle erreichbaren Zustände) fallen zusammen; steady-state (quasistationär) ist kleiner. Das zyklische Maß (Shiryaev) ist nicht gerechnet.

| δ | einfache Karte (gleiche ARL₀) | zero-state = Pollak = Lorden | steady-state | steady / zero | Regeln / einfach (zero, steady) |
|---|---|---|---|---|---|
| 0,5 | 46,42 | 27,33 | 25,93 | 0,949 | 0,589; 0,559 |
| 1 | 16,32 | 9,22 | 8,54 | 0,926 | 0,565; 0,523 |
| 2 | 3,418 | 3,128 | 2,948 | 0,943 | 0,915; 0,862 |
| 3 | 1,481 | 1,667 | 1,628 | 0,977 | 1,125; 1,099 |

**Das Maß dreht hier die Rangfolge nicht** (vier Regeln dieser Fassung): in jedem der vier δ liegt das Verhältnis zur einfachen Karte in beiden gerechneten Maßen auf derselben Seite von 1 (der Plan nannte 0,863 statt 0,862: Rundung der Zwischenwerte). Für die einfache Karte sind alle vier Maße gleich 1/p(δ). Die Deutung „eine Vorgeschichte kann den Zählern einer Regel nur etwas hinzufügen, nie einen Alarm verhindern, deshalb ist der Start ohne Vorgeschichte der schlechteste Zustand“ ist nicht bewiesen, nur durch die Messung gestützt.

**E. Fassungsfalle: Regel 4 mit 8, 9 oder 10 in Folge** (vier Regeln, ARL₀ / ARL₁ bei δ = 1): 8 in Folge 91,75 / 9,22; 9 in Folge 109,05 / 9,70; 10 in Folge 120,08 / 10,05; ohne Regel 4 132,89 / 10,95. Wer „die vier Regeln“ sagt, ohne die Fassung zu nennen, nennt eine ARL₀, die je nach Fassung (8, 9 oder 10 in Folge) zwischen 91,75 und 120,08 liegt (Faktor bis 1,31). Welche Fassung eine bestimmte Quelle meint, ist nicht gelesen.

**F. Alarme je Fenster zu 1000 Beobachtungen mit Neustart (exakt).** Einfache Karte ±3σ: Binomial(1000, p), Mittel 2,70; P(0 Alarme) = 6,70 %, P(≥ 5) = 13,68 %, P(≥ 8) = 0,65 %; Quantile 5 % / 50 % / 95 %: 0 / 3 / 6.
Zusatzregeln (Erneuerungsrechnung): Mittel 4,433 / 7,512 / 10,866 für 1+2 / 1+2+3 / 1+2+3+4 (W/ARL₀ = 4,436 / 7,525 / 10,899); P(0 Alarme) = 1,163 / 0,048 / 0,001 %; P(≥ 5) = 45,56 / 87,31 / 98,63 %; Median 4 / 7 / 11, 95-%-Quantil 8 / 12 / 16.

**G. Ausblick (hier nicht gerechnet):** einseitig, ARL₀ = 370, δ = 1: einfache Karte 26,746 Beobachtungen, ein CUSUM mit k = 0,5 braucht 8,573 (Zahl aus der Konzepte-Reihe, `cusum-gegen-karte-demo`, nicht aus dieser Demo; Messung der Konzepte-Reihe, nicht im Test).

## Zahl → Test (Band oder exakt)

Jede Zahl der README steht in `tests/test_claims.py` (die Datei prüft, dass die genannte Schreibweise in dieser README vorkommt und mit der Rechnung übereinstimmt) oder ist als „Messung dieser README, nicht im Test“ gekennzeichnet.

| Zahlen | Test | Art |
|---|---|---|
| Abschnitt A: 370,398; 740,80; 2,9997; 2,7818; 19 / 257 / 1109 / 1704; 1107 / 1702; 23,69 %; 23,71 %; 23,50 %; 23,48 %; 6,70 %; ARL₀-Tabelle; c-Reihe | `test_claims.py::test_abschnitt_a_*`, `test_handbeispiele.py` (Ganzzahl und Bruch) | exakt |
| Abschnitt B: ARL₀, Median, Quantile, Anteile, Abstände | `test_claims.py::test_abschnitt_b_*`, `test_oracle_karte.py` (Fensterkette, Aufzählung 8^6 und 8^8) | exakt |
| Abschnitt C: Tabelle, Schwellen, Kreuzungen, Verzögerungsverteilung, ±3σ-Tabelle | `test_claims.py::test_abschnitt_c_*` | exakt |
| Abschnitt D: Maße der Verzögerung | `test_claims.py::test_abschnitt_d_*`, `test_oracle_karte.py::test_delay_measures_order_for_all_rule_sets` | exakt |
| Abschnitt E: Fassungen von Regel 4 | `test_claims.py::test_abschnitt_e_*` | exakt |
| Abschnitt F: Alarme je Fenster | `test_claims.py::test_abschnitt_f_*`, `test_oracle_karte.py` (Erneuerung ↔ Binomial) | exakt |
| Zustandszahlen, Befunde und Korrekturen (28 %; 0,3 % und 1,5 %; 12 Zellen) | `test_claims.py::test_befunde_und_korrekturen`, `test_oracle_karte.py::test_state_counts_of_both_chains_are_documented` | exakt |
| Handbeispiele (38,6164; 118900/3079; 199/10000; 27501/1000000; 0,2371) | `test_handbeispiele.py` | exakt (Brüche) |
| Preset-Hilfetexte und Verdikt-Zeilen | `test_presets.py`, `test_app.py::test_each_preset_shows_its_expected_verdict`, `test_verdicts.py` | exakt |
| Literaturwerte (NIST, Jones-Farmer u. a., Lucas/Saccucci) | `test_literatur_karte.py` | exakt, Toleranz eine Einheit der gedruckten Stelle |
| Simulation: Mittel und Anteile im Band | `test_oracle_karte.py::test_simulated_*` (4,5 Standardfehler, zwei Zufalls-Seeds) | Band |
| Zahlen der Abschnitte „Befunde und Korrekturen“ zu den Simulationen (366,5; 382,5; 21,87 %; 6,20 %; 8,00 %; 2,739; 2,610) | – | Messung dieser README, nicht im Test |
| Abschnitt G (26,746) | `test_claims.py::test_abschnitt_g_ausblick_einseitig` (nur die einfache Karte); 8,573 gehört der Konzepte-Reihe | exakt, 8,573 nicht hier gemessen |
| Rechenzeiten | – | Messung dieser README, nicht im Test (keine Zeit-Assertion) |

## 🚧 Ehrliche Grenzen: bewusst nicht enthalten und nicht gemessen

| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Ziel und Streuung bekannt** | Eine Karte mit aus Daten geschätztem Ziel und geschätzter Streuung hat eine andere Lauflänge als die hier gerechnete. | nicht gemessen |
| **Beobachtungen unabhängig und normalverteilt** | Abhängigkeit oder Schiefe verändern ARL₀ und Verzögerung. | nicht gemessen |
| **Ein Kanal** | Mehrere Messreihen gleichzeitig zu überwachen ist eine andere Aufgabe. | nicht gemessen |
| **Zusatzregeln in genau dieser Fassung** („nach Art der Western-Electric-Regeln“, Regel 4 mit 8 in Folge) | Mit 9 in Folge steigt die ARL₀ von 91,75 auf 109,05: wer „die vier Regeln“ sagt, ohne die Fassung zu nennen, nennt eine ARL₀, die je nach Fassung (8, 9 oder 10 in Folge) zwischen 91,75 und 120,08 liegt (Faktor bis 1,31). | gemessen (Experiment 5) |
| **Neustart nach Alarm ohne Vorgeschichte** | Die Alarmzahl je Fenster gilt nur mit dieser Festlegung; ARL ist nur sinnvoll, wenn bei Alarm angehalten wird. | Festlegung dieser Seite |
| **Zähleinheit zero-state** (Start ohne Vorgeschichte) | Für die Zusatzregeln ist die Verzögerung im quasistationären steady-state kleiner (bei δ = 1: 8,54 gegen 9,22 Beobachtungen); das zyklische Maß (Shiryaev) ist hier nicht gerechnet. | steady-state gemessen, Shiryaev nicht |
| **Ausreißer gegen Verschiebung** | Die einfache Karte meldet einen einzelnen Ausreißer, eine Karte mit Gedächtnis meldet vor allem Verschiebungen; ob Ausreißer oder Verschiebung gesucht wird, ist die Frage der Anomalie-Erkennung. | nicht Gegenstand |
| **Gedächtnis der Karte** | Bei einseitiger Karte, ARL₀ = 370 und Sprung δ = 1 braucht die einfache Karte 26,746 Beobachtungen, ein CUSUM (k = 0,5) 8,573: dieser Wert ist eine Zahl aus der Konzepte-Reihe (cusum-gegen-karte-demo), hier nicht gerechnet. | Ausblick, nicht hier gemessen |
| **Nutzen im Betrieb** | Ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist hier nicht belegt. | nicht belegt |

Weiter nicht enthalten: einseitige Fassungen der Regeln 2 bis 4 (nicht definiert und nicht gemessen; die App bietet „einseitig“ nur für Regel 1 allein an), andere Regelkombinationen (nur Regel 4, 1+4), andere Fassungen der Regeln nach einer Quelle, der Kreuzungspunkt in steady-state, der Linux-Lauf der Tests (Zahlen und Zeiten nur unter Windows, Python 3.14.6), ein PDF-Export. Die Zahlen sind unter Windows gemessen; Ganzzahlen, Brüche und Ketten sind plattformfest bis auf die Rundung der letzten Stelle, Simulationen sind Bänder.

## Quellen (Stufe der Lesung)

Alle Gleichungen und Befunde stehen in eigenen Worten da; es sind keine Absätze oder Tabellen der Quellen übernommen. Die Stufe „gelesen“ meint die Lesung bei der Planung der Konzepte-Reihe, im Bau nicht erneut gelesen. Alles hier ist Lehrbuchwissen; neu ist nur die Vorführbarkeit mit Zähleinheit.

* NIST/SEMATECH e-Handbook of Statistical Methods, 6.3.1 und 6.3.2.1: Grenzen ±3σ, Einseitenwahrscheinlichkeit 0,00135, ARL = 1/p, etwa 371. **Gelesen** über die Zusammenfassung eines Abrufwerkzeugs, nicht als Roh-HTML; keine Tabellenzahl des Handbuchs wird übernommen.
* Lucas/Saccucci 1990, Technometrics 32(1), 1–12: Lauflängen „nahezu geometrisch“; Tabelle 3, Spalte λ = 1 (L = 3,090, ARL₀ = 500, Shewhart-ARL 54,6; 17,9; 7,26 bei δ = 1; 1,5; 2, eigene Nachrechnung 54,55; 17,88; 7,25). **Gelesen**, die Kopie hat Zeichenverluste.
* Jones-Farmer/Woodall/Steiner/Champ 2014, Journal of Quality Technology 46(3), 265–280: FAP = 1 − (1 − p)^m, 0,078 bei m = 30 und 0,126 bei m = 50. **Gelesen** (Abschnitt 2).
* Zwetsloot/Jones-Farmer/Woodall 2024, Quality Engineering 36(3), 487–499: ARL nur sinnvoll, wenn bei Alarm angehalten wird. **Gelesen.** Der Neustart nach Alarm ist die Festlegung dieser Seite, nicht der Quelle.
* **Nicht gelesen:** Shewhart 1924/1931, Montgomery, das Western-Electric-Handbuch, Champ/Woodall 1987 (die ARL₀ „etwa 92“ der vier Regeln stand nur im Gedächtnis; die Rechnung gibt 91,7508 mit zwei verschieden gebauten Ketten, die Aufzählung bestätigt die ersten acht Wahrscheinlichkeiten; das ist Rechnung, keine Quellenbestätigung), Basseville/Nikiforov, Page 1954.

## Verifikation

Die Suite hat 226 Tests und läuft seriell in 99 s und 121 s bei zwei Läufen (Windows, ein Thread, Maschine zeitweise belastet, Messung dieser README; der einzelne langsame Test ist die Fensterkette für Regel 4 mit 9 in Folge mit rund 10 s).

* **Kern:** Handbeispiele als exakte Tests (Brüche, Ganzzahlen); Orakel-Tests (geschlossene Form, Fensterkette, Aufzählung aller Zonenfolgen, Schleifen-Referenz der Regeln, Erneuerung ↔ Binomial, Simulation im Band); Eigenschaften (Monotonie, Spiegelsymmetrie, Schachtelung der Regelsätze, Skalierung der einfachen Karte); Randfälle (c ≤ 0, sehr großes c, δ = 0 und sehr großes δ, Regel 4 mit weniger als 8 Beobachtungen, ν = 0 und ν hinter der Reihe, Fenster 1, ein Lauf).
* **Claims:** jede Zahl der Abschnitte A bis G gegen die Rechnung (`tests/test_claims.py`); Verdikte als reine Funktionen mit handgesetzten Werten (`tests/test_verdicts.py`); je Schnellstart-Szenario die erwartete Verdikt-Zeile (ein Szenario ohne Sprung meldet keine Verzögerung).
* **Oberfläche:** AppTest-Rauchtest (jeder Knopf, jeder Regler an Minimum und Maximum, alle Regelsätze und Fassungen von Regel 4, Eckfall-Matrix), Layout (Schnellstart oben, fester Kopf, aktives Szenario hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
* **Eingefrorene Aufzählung:** `lfk_enum.json` (P(T = t), t = 1 … 8, aus allen 8^8 Zonenfolgen, Quelle und Stand je Feld); `python tools/lfk_make_enum.py --check` rechnet neu (rund 30 s) und vergleicht.
* **Nicht geprüft:** Linux-Lauf der Tests, die App in der Streamlit Cloud, die Anzeige in einem echten Browser außerhalb der Vorschau.

## Rechenzeit (Messung dieser README, nicht im Test)

Entwicklungsmaschine, Windows, ein Thread, Hinweise ohne Gewähr für die Cloud. Kette für vier Regeln bauen 0,03 s, lösen 0,02 s, Verteilung 0,05 s; Vergleich bei gleicher ARL₀ (Kreuzungssuche) je Regelsatz bis 0,4 s; Alarme je Fenster bei W = 5000 mit vier Regeln 0,09 s; Einzelreihe mit `normal()` bei 1000 Beobachtungen 0,07 s.
Simulation auf Klick: einfache Karte 5000 Läufe bei ARL₀ = 370,4 (1,85 Mio. Beobachtungen) 1,4 s; vier Regeln 5000 Läufe 1,8 s; 2000 Fenster zu 1000 Beobachtungen (einfache Karte) 1,6 s, vier Regeln 600 Fenster 2,3 s. Die Obergrenze der Simulation liegt bei 2 Mio. Beobachtungen je Klick (einfache Karte) bzw. 600 000 (Zusatzregeln); ist sie überschritten, wird die Zahl der Läufe begrenzt, und unter 100 Läufen wird nicht simuliert (Meldung statt eines Bands).
Erster Aufruf der App einschließlich Importen rund 4 s, jeder weitere Regleraufruf unter 1 s (Puffer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: eine Reihe durch die Karte, Kernfragen, fünf Experimente, Grenzen, Formeln |
| `lfk_chart.py` | Zonen, Alarmwahrscheinlichkeit und geschlossene Form der einfachen Karte, Regeln an Rohwerten |
| `lfk_chain.py` | Markov-Kette der Regelsätze: Verteilung der Lauflänge, ARL, steady-state, Pollak, Lorden |
| `lfk_renewal.py` | Alarme je Fenster (Erneuerungsrechnung, Binomial) |
| `lfk_scenario.py` | Einzelreihe mit Sprung, Simulation von Lauflängen und Fenstern |
| `lfk_rng.py` | SplitMix64 (Kopie der Vorlage des Website-Repos, nicht ändern) |
| `lfk_evaluation.py` | Einstellungen, Kennzahlen, Vergleich bei gleicher ARL₀, Verdikte (reine Funktionen) |
| `lfk_presets.py` | Permalink, Schnellstart-Szenarien (aktives hervorgehoben), Zufalls-Seed-Knopf, Schwelle für ARL₀ = 370 |
| `lfk_constants.py` | Regler-Grenzen, Rechenbudget, Farben, Schnellstart-Szenarien |
| `lfk_texts.py` | Einleitung, Erklärung, Annahmen-Tabelle, Formeln (Annahmen-Tabelle gleich der README) |
| `lfk_layout.py` | Fester Kopf (Stylesheet) |
| `lfk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `lfk_enum.json` | eingefrorene Aufzählung aller 8^8 Zonenfolgen |
| `tools/lfk_make_enum.py` | erzeugt und prüft (`--check`) `lfk_enum.json` |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Handbeispiele, Orakel, Literatur, Eigenschaften, Randfälle, Claims, Verdikte, Schnellstart-Szenarien, Layout, AppTest, Zufallsstrom, Footer, Laufzeit-Importe (Testhilfen in `tests/helpers_lfk.py`) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Getestet wird mit Python 3.12 (CI) und 3.14 (lokal); die in der Streamlit Cloud eingestellte Version ist noch nicht festgelegt. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html).
