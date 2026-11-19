"""Feste Texte der App (Einleitung, Erklärung, Annahmen, Formeln). Zahlen in diesen Texten prüft tests/test_claims.py;
Zahlen, die sich mit den Reglern ändern, kommen nicht von hier, sondern aus der Rechnung (lfk_evaluation.verdict_*)."""

INTRO = """
Erste Seite der Reihe **Änderungserkennung: wie schnell merkt man, dass sich etwas geändert hat?** Eine **Kontrollkarte** schlägt **Alarm**, sobald die Beobachtungen die **Schwelle** der Karte überschreiten.
Die oft genannte Zahl **ARL₀ = 370** ist der *Mittelwert* der **Lauflänge** (Beobachtungen bis zum Fehlalarm): wie diese Wartezeit **streut**, und wie lange die Karte nach einem **Sprung** des Mittelwerts braucht,
rechnet diese Demo exakt vor und prüft es gegen Simulation. Alles gilt für **unabhängige Normalwerte mit bekanntem Ziel und bekannter Streuung**; gezählt wird in **Beobachtungen**, **Reihen** und **Fenstern**.

Weiter unten: die **Kernfragen**, fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung** mit Lesehilfe.
"""

HOW_IT_WORKS = """
1. **Beobachten:** Jede Beobachtung $x_t$ wird in σ-Einheiten gegen das Ziel 0 gelesen. Vor der Änderung ist $x_t \\sim N(0, 1)$, danach $N(\\delta, 1)$.
2. **Regel anwenden:** Die **einfache Karte** (Regel 1) schlägt an, wenn $|x_t| > c$ ist (zweiseitig; einseitig: $x_t > c$). Die **Zusatzregeln** lesen dazu bis zu zehn Beobachtungen zurück (Regeln 2 bis 4, Fassung unten).
3. **Zählen:** Die **Lauflänge** $T$ ist die Zahl der Beobachtungen bis einschließlich der alarmierenden. **ARL₀** ist ihr Mittel ohne Änderung, **ARL₁** das Mittel nach einem Sprung.
4. **Warum die Zahl täuscht:** Die einfache Karte vergisst alles (Gedächtnis null), ihre Lauflänge ist **geometrisch** verteilt: viele frühe Alarme, ein langer rechter Rand; der Median liegt bei etwa 69 % des Mittels (ln 2).
"""

# Annahmen und Grenzen: dieselbe Tabelle steht in der README (tests/test_claims.py vergleicht beide Fassungen).
ASSUMPTIONS_MARKDOWN = """
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
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** *Gegeben:* unabhängige Beobachtungen $x_1, x_2, \dots$ in σ-Einheiten mit bekanntem Ziel 0 und bekannter Streuung 1; ein Sprung des Mittelwerts um $\delta$ nach $\nu$ Beobachtungen; eine Alarmregel (Schwelle $c$ oder Regelsatz).
*Gesucht:* die Verteilung der Lauflänge $T$ bis zum Alarm, ihr Mittel (ARL₀ ohne, ARL₁ mit Sprung), die Verzögerung $D$ und die Alarmzahl in einem Fenster.

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Beobachtung $t$ in σ-Einheiten |
| $\delta$ | Sprung des Mittelwerts in σ-Einheiten ($\delta = 0$: keine Änderung) |
| $\nu$ | Zahl der Beobachtungen vor der Änderung (letzte unveränderte Beobachtung; $\nu = 0$: die Änderung liegt vor der ersten) |
| $c$ | Schwelle der einfachen Karte in σ-Einheiten |
| $T$ | Lauflänge: Beobachtungen bis einschließlich der alarmierenden ($T \ge 1$) |
| $\text{ARL}_0,\ \text{ARL}_1(\delta)$ | $E[T]$ ohne Änderung bzw. nach einem Sprung der Größe $\delta$ |
| $D$ | Verzögerung $D = T - \nu \ge 1$ (der Alarm bei der ersten veränderten Beobachtung zählt 1) |
| $p(\delta)$ | Alarmwahrscheinlichkeit je Beobachtung der einfachen Karte |
| $Z_t$ | Zone der Beobachtung: $0$: $x < -3$; $1$: $-3 \dots -2$; $2$: $-2 \dots -1$; $3$: $-1 \dots 0$; $4$: $0 \dots 1$; $5$: $1 \dots 2$; $6$: $2 \dots 3$; $7$: $x > 3$ |
| $R_\delta$ | Übergangsmatrix der Karte ohne Alarm bei Sprung $\delta$ |
| $\mathbf{e}_0$ | Startzustand ohne Vorgeschichte (zero-state) |
| $W$ | Fensterlänge in Beobachtungen (Alarme je Fenster, Neustart nach Alarm) |
| $N_W$ | Zahl der Alarme in einem Fenster der Länge $W$ |
| $S_k$ | Summe von $k$ unabhängigen Lauflängen |
| $t_0$ | Fensterlänge für „Alarm innerhalb der ersten $t_0$ Beobachtungen“ |

**Beobachtungen.** $x_t \sim N(0, 1)$ für $t \le \nu$ und $x_t \sim N(\delta, 1)$ für $t > \nu$, alle unabhängig.

**Regeln** (Fassung dieser Demo, nach Art der Western-Electric-Regeln; die Fassungen der Literatur unterscheiden sich). Regel 1: $|x_t| > 3$ (einfache Karte mit Schwelle $c$: $|x_t| > c$, einseitig $x_t > c$).
Regel 2: zwei von drei aufeinanderfolgenden Beobachtungen jenseits $2$ auf derselben Seite. Regel 3: vier von fünf jenseits $1$ auf derselben Seite. Regel 4: $L$ Beobachtungen in Folge auf derselben Seite der Mittellinie ($L = 8, 9, 10$).
Der Regelsatz „1+2+3+4“ schlägt an, sobald eine der vier Regeln anschlägt; die Regelsätze sind geschachtelt.

**Einfache Karte.** Mit $p(\delta) = \Phi(\delta - c) + \Phi(-c - \delta)$ (zweiseitig; einseitig nur der erste Summand):

$$P(T > t) = (1 - p)^t, \qquad \text{ARL} = \frac{1}{p}$$

Die Lauflänge ist exakt geometrisch verteilt (gedächtnislos); das $q$-Quantil ist das kleinste $t$ mit $1 - (1-p)^t \ge q$.

**Zusatzregeln als Markov-Kette.** Der Zustand ist die Folge der Zonen der letzten Beobachtungen (bis zu vier) und die Länge des Laufs auf einer Seite der Mittellinie (nur mit Regel 4). Eine Beobachtung, bei der keine Regel anschlägt, führt in den Folgezustand; $R_\delta$ enthält deren Wahrscheinlichkeiten $P(Z = z)$ aus $N(\delta, 1)$:

$$P(T > t) = \mathbf{e}_0^\top R_\delta^{\,t}\, \mathbf{1}, \qquad \text{ARL} = \mathbf{e}_0^\top (I - R_\delta)^{-1} \mathbf{1}$$

Die Zonengrenzen sind die Regelgrenzen, deshalb gibt es keinen Diskretisierungsfehler.

**Verzögerung und Maße.** $D = T - \nu$. zero-state: Start in $\mathbf{e}_0$. steady-state (quasistationär): Start in der Grenzverteilung der alarmfreien Kette bei $\delta = 0$ (Grenzwert von $E[D]$ für $\nu \to \infty$, bedingt auf „kein Alarm davor“; nicht das Shiryaev-Maß).
Pollak: $\max_\nu E[T - \nu \mid T > \nu]$. Lorden: Maximum über alle erreichbaren Startzustände der mittleren Verzögerung.

**Alarme je Fenster.** Nach jedem Alarm Neustart ohne Vorgeschichte (Festlegung dieser Seite). Mit $S_k$ als Summe von $k$ unabhängigen Lauflängen:

$$P(N_W \ge k) = P(S_k \le W)$$

Für die einfache Karte ist $N_W$ binomialverteilt mit den Parametern $W$ und $p(0)$.

*Der Block ist ein vollständiges Modell der Karte und ihrer Zähler; was fehlt, ist die Wirklichkeit außerhalb der Annahmen (siehe 🚧).*

**Quellen.** NIST/SEMATECH e-Handbook of Statistical Methods, Abschnitte 6.3.1 und 6.3.2.1 (Grenzen ±3σ, ARL = 1/p, etwa 371; gelesen über die Zusammenfassung eines Abrufwerkzeugs, keine Tabellenzahl übernommen).
Lucas/Saccucci 1990, Technometrics 32(1), 1–12 (Lauflängen nahezu geometrisch; Tabelle 3, Spalte λ = 1; Kopie mit Zeichenverlusten gelesen).
Jones-Farmer/Woodall/Steiner/Champ 2014, Journal of Quality Technology 46(3), 265–280 (FAP = 1 − (1 − p)^m). Zwetsloot/Jones-Farmer/Woodall 2024, Quality Engineering 36(3), 487–499 (ARL nur sinnvoll, wenn bei Alarm angehalten wird).
Nicht gelesen: Shewhart 1924/1931, Montgomery, das Western-Electric-Handbuch, Champ/Woodall 1987.
"""
