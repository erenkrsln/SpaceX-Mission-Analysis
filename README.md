# 🚀 SpaceX Mission Analysis – Data Science & Machine Learning

Willkommen bei meinem **SpaceX Mission Analysis Projekt**!

In diesem Projekt wird eine Python-basierte Anwendung zur Analyse historischer SpaceX-Raketenstarts entwickelt. Ziel ist es, mithilfe von **Datenanalyse, Datenvisualisierung und Machine Learning** relevante Einflussfaktoren zu untersuchen, Trends zu erkennen und erfolgreiche Raketenstarts vorherzusagen.

Das Projekt verbindet die automatisierte Datenerfassung über eine REST-API mit der Verarbeitung und Visualisierung von Daten sowie der Implementierung eines Machine-Learning-Modells.

## 📊 Hintergrund und Kontext

SpaceX gehört zu den führenden Unternehmen der kommerziellen Raumfahrt und setzt auf innovative Technologien zur Verbesserung der Zuverlässigkeit und Wirtschaftlichkeit von Raketenstarts.

Durch die Analyse historischer Missionsdaten lassen sich Entwicklungen, Erfolgsquoten und mögliche Zusammenhänge zwischen verschiedenen Einflussfaktoren untersuchen.

**Zentrale Fragestellung:**

*Wie können historische SpaceX-Missionsdaten genutzt werden, um erfolgreiche Raketenstarts zu analysieren und mithilfe von Machine Learning vorherzusagen?*

## 🔍 Projektablauf

**1. Datenbeschaffung**

- Automatisierter Abruf von SpaceX-Missionsdaten über eine öffentliche REST-API.
- Verarbeitung von JSON-Daten und Speicherung in CSV-Dateien.

**2. Datenaufbereitung**

- Einlesen und Strukturieren der Datensätze mit Python und Pandas.
- Bereinigung und Vorbereitung der Daten für weitere Analysen.

**3. Explorative Datenanalyse (EDA)**

- Untersuchung historischer Raketenstarts und ihrer Erfolgsquoten.
- Analyse zeitlicher Entwicklungen und Trends.
- Erstellung von Diagrammen mit Matplotlib.

**4. Machine Learning**

- Implementierung eines Klassifikationsmodells mittels logistischer Regression.
- Aufbereitung relevanter Merkmale für das Modelltraining.
- Klassifikation erfolgreicher und fehlgeschlagener Raketenstarts.

**5. Softwarequalität und Testing**

- Strukturierte Organisation des Python-Projekts.
- Implementierung automatisierter Unit-Tests.
- Dokumentation der Installation und Ausführung.

## 🛠️ Verwendete Technologien

| Technologie | Einsatzbereich |
|---|---|
| Python | Softwareentwicklung und Datenanalyse |
| Pandas | Datenverarbeitung und Datenbereinigung |
| Matplotlib | Datenvisualisierung |
| Scikit-Learn | Machine Learning |
| REST API | Abruf historischer SpaceX-Daten |
| JSON & CSV | Datenverarbeitung und Datenspeicherung |
| Unittest | Automatisierte Tests |
| Git & GitHub | Versionsverwaltung |

## 🎯 Ziel und Nutzen

Ziel dieses Projekts ist es, praktische Methoden aus den Bereichen **Softwareentwicklung, Data Science und Machine Learning** miteinander zu verbinden.

Das Projekt demonstriert, wie Daten automatisiert erfasst, systematisch analysiert und mithilfe statistischer Modelle ausgewertet werden können.

Die eingesetzten Methoden lassen sich grundsätzlich auch auf andere datengetriebene Anwendungsbereiche übertragen, beispielsweise die Analyse von Produktionsdaten, Qualitätskennzahlen und technischen Prozessen.

## 📁 Projektstruktur

Das Repository ist in verschiedene Bereiche unterteilt:

- **`src/`** – Python-Quellcode für Datenbeschaffung und Analyse
- **`data/`** – Beispieldaten und heruntergeladene SpaceX-Datensätze
- **`tests/`** – Automatisierte Unit-Tests
- **`outputs/`** – Generierte Analysen und Visualisierungen
- **`requirements.txt`** – Benötigte Python-Bibliotheken

## 🚀 Ausführung

**1. Repository klonen**

```bash
git clone https://github.com/erenkrsln/SpaceX-Mission-Analysis.git
cd SpaceX-Mission-Analysis
```

**2. Abhängigkeiten installieren**

```bash
python -m pip install -r requirements.txt
```

**3. Datenanalyse ausführen**

```bash
python -m src.analysis --input data/example_launches.csv
```

**4. Automatisierte Tests starten**

```bash
python -m unittest discover -s tests -v
```

## 🔮 Mögliche Weiterentwicklungen

- Integration zusätzlicher Machine-Learning-Algorithmen wie KNN, SVM und Decision Trees.
- Erweiterte Modellbewertung durch Accuracy, Precision, Recall und F1-Score.
- Integration einer SQL-Datenbank zur Speicherung und Abfrage von Missionsdaten.
- Entwicklung eines interaktiven Dashboards mit Streamlit.
- Erweiterung der Analyse um erfolgreiche Landungen wiederverwendbarer Raketenstufen.

## 📚 Projektkontext

Das Projekt orientiert sich thematisch an Fragestellungen aus dem Bereich Data Science und an den Inhalten des IBM Data Science Professional Certificate.

Die Anwendung wurde als eigenständiges Python-Projekt mit Fokus auf Datenbeschaffung, Analyse, Visualisierung und Machine Learning aufgebaut. Bei der Erstellung wurden KI-gestützte Entwicklungswerkzeuge verwendet.

## 👨‍💻 Autor

**Eren Karaarslan**  
Media Engineering – Technische Hochschule Nürnberg

**GitHub:** https://github.com/erenkrsln