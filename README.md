# 🚀 SpaceX Mission Analysis – Data Science & Machine Learning

Willkommen bei meinem **SpaceX Mission Analysis Projekt**!

In diesem Projekt wird eine Python-basierte Anwendung zur Analyse und Vorhersage erfolgreicher Landungen der ersten Stufe von **SpaceX Falcon 9 Raketen** entwickelt.

Ziel ist es, mithilfe von **Datenanalyse, Datenvisualisierung und Machine Learning** historische Missionsdaten auszuwerten, Einflussfaktoren auf den Landungserfolg zu untersuchen und Vorhersagemodelle zu entwickeln.

Das Projekt kombiniert die automatisierte Datenbeschaffung, strukturierte Datenverarbeitung, explorative Datenanalyse (EDA) und prädiktive Modellierung mit Python und Scikit-Learn.

## 📊 Hintergrund und Kontext

SpaceX setzt auf wiederverwendbare Raketenstufen, um die Wirtschaftlichkeit von Raumfahrtmissionen zu verbessern. Ein entscheidender Bestandteil dieses Konzepts ist die erfolgreiche Landung und Wiederverwendung der ersten Raketenstufe.

Durch die Analyse historischer Falcon-9-Missionen lassen sich Muster erkennen und mögliche Zusammenhänge zwischen technischen Eigenschaften und dem Erfolg einer Landung untersuchen.

**Zentrale Fragestellung:**

*Wie können historische Falcon-9-Missionsdaten verwendet werden, um den Erfolg einer Erststufenlandung mithilfe von Machine Learning vorherzusagen?*

## 🔍 Projektablauf

### 1. Datenbeschaffung

- Import historischer SpaceX-Missionsdaten aus einem archivierten JSON-Datensatz der öffentlichen SpaceX-API.
- Automatisierte Verarbeitung und Speicherung der Missionsdaten als CSV-Dateien.
- Filterung relevanter Falcon-9-Missionen mit dokumentierten Erststufen-Landeversuchen.

### 2. Datenaufbereitung

- Einlesen, Bereinigen und Strukturieren der Datensätze mithilfe von Python und Pandas.
- Behandlung fehlender Werte und Umwandlung kategorialer beziehungsweise boolescher Merkmale.
- Auswahl relevanter Eigenschaften für die Modellierung:
  - Anzahl bisheriger Booster-Flüge
  - Wiederverwendung der Raketenstufe
  - Verwendung von Grid Fins
  - Verwendung von Landebeinen
- Aufbereitung der Zielvariable `landing_success` zur Unterscheidung zwischen erfolgreichen und fehlgeschlagenen Landungen.

### 3. Explorative Datenanalyse (EDA)

- Untersuchung von 153 historischen Falcon-9-Landeversuchen aus den Jahren 2013 bis 2022.
- Analyse der jährlichen Erfolgsquoten.
- Darstellung zeitlicher Entwicklungen und Trends.
- Erstellung von Diagrammen mit Matplotlib.
- Export der Analyseergebnisse in CSV-Dateien.

### 4. Machine Learning

- Entwicklung eines Klassifikationsmodells mittels logistischer Regression.
- Implementierung einer Preprocessing-Pipeline mit Scikit-Learn.
- Aufteilung der Daten in Trainings- und Testdatensätze.
- Berücksichtigung der ungleichen Klassenverteilung durch `class_weight="balanced"`.
- Vergleich mit einem einfachen Baseline-Modell.
- Evaluation anhand von Accuracy, Balanced Accuracy, Precision, Recall, F1-Score und Confusion Matrix.

### 5. Softwarequalität und Testing

- Modularer Aufbau der Python-Anwendung.
- Trennung von Datenbeschaffung, Datenanalyse und Machine Learning.
- Implementierung automatisierter Unit-Tests für die ursprüngliche Analysekomponente.
- Verwendung von Git und GitHub zur Versionsverwaltung.
- Dokumentation der Installation und Ausführung.

## 🛠️ Verwendete Technologien

| Technologie | Einsatzbereich |
|---|---|
| Python | Softwareentwicklung und Datenanalyse |
| Pandas | Datenbereinigung und Datenverarbeitung |
| Matplotlib | Erstellung von Diagrammen |
| Scikit-Learn | Machine Learning und Modellbewertung |
| Logistic Regression | Klassifikation von Landungsergebnissen |
| REST API / JSON | Datenbeschaffung und Datenverarbeitung |
| CSV | Strukturierte Speicherung von Missionsdaten |
| Unittest | Automatisierte Softwaretests |
| Git & GitHub | Versionsverwaltung |

## 📈 Ergebnisse der Datenanalyse

Für die Untersuchung wurden **153 historische Falcon-9-Landeversuche** aus einem archivierten SpaceX-Datensatz berücksichtigt.

| Kennzahl | Ergebnis |
|---|---:|
| Analysierte Landeversuche | 153 |
| Erfolgreiche Landungen | 142 |
| Fehlgeschlagene Landungen | 11 |
| Erfolgsquote | 92,8 % |
| Zeitraum | 2013–2022 |

Die Analyse zeigt, dass die Erfolgsquote der Falcon-9-Erststufenlandungen im betrachteten Zeitraum deutlich gestiegen ist.

Gleichzeitig besteht eine starke Ungleichverteilung zwischen erfolgreichen und fehlgeschlagenen Landungen. Diese stellt eine besondere Herausforderung für die Entwicklung zuverlässiger Klassifikationsmodelle dar.

## 🤖 Machine-Learning-Ergebnisse

Zur Vorhersage erfolgreicher Landungen wurde eine **logistische Regression** implementiert und mit einem einfachen Baseline-Modell verglichen.

Die Daten wurden stratifiziert in 107 Trainings- und 46 Testdatensätze aufgeteilt.

### Modellvergleich

| Metrik | Baseline | Logistische Regression |
|---|---:|---:|
| Accuracy | 93,5 % | 69,6 % |
| Balanced Accuracy | 50,0 % | 83,7 % |
| Recall – Fehlgeschlagene Landungen | 0 % | 100 % |
| Precision – Fehlgeschlagene Landungen | 0 % | 18 % |
| F1-Score – Fehlgeschlagene Landungen | 0,00 | 0,30 |

### Interpretation

Das Baseline-Modell erreicht eine hohe Accuracy, da es sämtliche Landeversuche als erfolgreich klassifiziert.

Die logistische Regression erkennt hingegen alle drei fehlgeschlagenen Landungen im Testdatensatz, erzeugt dabei jedoch 14 Fehlalarme.

Die Ergebnisse verdeutlichen, dass die reine Accuracy bei unausgeglichenen Datensätzen keine ausreichende Bewertungsmetrik darstellt.

Die höhere Balanced Accuracy der logistischen Regression weist auf eine bessere Erkennung beider Klassen in dieser konkreten Testaufteilung hin.

Aufgrund der geringen Anzahl fehlgeschlagener Landungen und der zufälligen Trainings-/Testaufteilung sind die Ergebnisse jedoch mit erheblicher Unsicherheit verbunden. Das Modell ist nicht als zuverlässige Vorhersage für zukünftige Raumfahrtmissionen validiert.

### Confusion Matrix

Die Confusion Matrix zeigt, welche Landungen korrekt und welche falsch klassifiziert wurden.

| Tatsächliches Ergebnis | Vorhergesagt: Fehlschlag | Vorhergesagt: Erfolg |
|---|---:|---:|
| Fehlgeschlagene Landung | 3 | 0 |
| Erfolgreiche Landung | 14 | 29 |

Die grafische Darstellung wird im Ordner `outputs/landings/` gespeichert.

## 🎯 Ziel und Nutzen

Dieses Projekt demonstriert die praktische Anwendung von Methoden aus den Bereichen **Softwareentwicklung, Data Science und Machine Learning**.

Im Mittelpunkt stehen:

- Automatisierte Erfassung und strukturierte Verarbeitung technischer Daten.
- Explorative Datenanalyse zur Identifikation relevanter Entwicklungen.
- Implementierung und Bewertung eines Machine-Learning-Klassifikationsmodells.
- Kritische Interpretation statistischer Ergebnisse.
- Visualisierung datenbasierter Erkenntnisse.

Die verwendeten Methoden können grundsätzlich auch auf andere technische Anwendungsgebiete übertragen werden, beispielsweise auf die Analyse von Produktionsdaten, Maschinenzuständen oder Qualitätskennzahlen in industriellen Prozessen.

## 📁 Projektstruktur

Das Repository ist modular aufgebaut:

```text
SpaceX-Mission-Analysis/
├── data/
│   ├── example_launches.csv
│   ├── spacex_historical.csv
│   └── falcon9_landings.csv
│
├── src/
│   ├── fetch_data.py
│   ├── import_snapshot.py
│   ├── import_landings.py
│   ├── analysis.py
│   ├── model.py
│   ├── landing_analysis.py
│   └── landing_model.py
│
├── outputs/
│   └── landings/
│       ├── yearly_summary.csv
│       ├── landing_success_rate.png
│       └── confusion_matrix.png
│
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

Die vorhandenen Skripte zur Analyse allgemeiner Raketenstarts bleiben als ergänzende Projektbestandteile erhalten.

## 🚀 Installation und Ausführung

### 1. Repository klonen

```bash
git clone https://github.com/erenkrsln/SpaceX-Mission-Analysis.git
cd SpaceX-Mission-Analysis
```

### 2. Benötigte Bibliotheken installieren

```bash
python -m pip install -r requirements.txt
```

### 3. Historische Falcon-9-Landedaten importieren

```bash
python -m src.import_landings
```

Das Skript erstellt die Datei `data/falcon9_landings.csv`.

### 4. Explorative Datenanalyse durchführen

```bash
python -m src.landing_analysis
```

Die Analyse erstellt eine jährliche Übersicht und visualisiert die Erfolgsquoten historischer Erststufenlandungen.

### 5. Machine-Learning-Modell trainieren

```bash
python -m src.landing_model
```

Das Programm trainiert die logistische Regression, vergleicht sie mit einem Baseline-Modell und erstellt eine Confusion Matrix.

### 6. Automatisierte Tests ausführen

```bash
python -m unittest discover -s tests -v
```

Die bestehenden Unit-Tests prüfen Funktionen der ursprünglichen Datenanalyse.

## 🔮 Mögliche Weiterentwicklungen

- Integration zusätzlicher Klassifikationsmodelle wie KNN, SVM und Decision Trees.
- Hyperparameteroptimierung mit GridSearchCV.
- Zeitbasierte Modellvalidierung und Cross-Validation.
- Integration einer SQL-Datenbank für strukturierte Datenabfragen.
- Entwicklung eines interaktiven Dashboards mit Streamlit.
- Erweiterung um zusätzliche Missionsmerkmale wie Nutzlastmasse, Orbit und Startplatz.
- Ergänzung automatisierter Tests für die Landungsanalyse und das Vorhersagemodell.
- Vergleich unterschiedlicher Klassifikationsschwellen zur Reduzierung von Fehlalarmen.

## 📚 Projektkontext und Datenquelle

Das Projekt orientiert sich thematisch am **IBM Data Science Professional Certificate** und insbesondere an den dort behandelten Methoden zur Analyse und Vorhersage von SpaceX-Raketenlandungen.

Die Anwendung wurde als eigenständiges Python-Projekt mit Fokus auf Datenbeschaffung, Datenanalyse, Datenvisualisierung und Machine Learning aufgebaut. Bei der Erstellung wurden KI-gestützte Entwicklungswerkzeuge verwendet.

**Datenquelle:** Historischer JSON-Snapshot der öffentlichen SpaceX-API, bereitgestellt über das Open-Source-Repository [nisrulz/android-spacex-app](https://github.com/nisrulz/android-spacex-app).

Die Daten stellen einen archivierten historischen Stand dar und werden nicht automatisch um aktuelle SpaceX-Missionen ergänzt.

## 👨‍💻 Autor

**Eren Karaarslan**  
Media Engineering – Technische Hochschule Nürnberg

**GitHub:** https://github.com/erenkrsln