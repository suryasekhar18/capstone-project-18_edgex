# EdgeX Fog Computing Capstone

This project combines an iFogSim-based edge/fog simulation with two small
Python services for computational intelligence:

- **AI priority classifier** - compares a decision tree, random forest, and SVM
  and exposes the best model at `POST /predict_priority` on port `5000`.
- **Fuzzy resource allocator** - maps `LOW`, `MEDIUM`, or `HIGH` priority to
  bandwidth and CPU allocations at `POST /allocate_resources` on port `8080`.
- **iFogSim simulation** - Java simulations and datasets under [`iFogSim/`](./iFogSim/).

## Project layout

```text
computational_intelligence_part/
  ai_classifier_server.py
  fuzzy_allocator_server.py
  priority_dataset.csv
  requirements.txt
iFogSim/
  src/       Java simulation source
  dataset/   mobility and resource datasets
  results/   generated reports and spreadsheets
```

## Quick start

### 1. Start the Python services

From `computational_intelligence_part`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python ai_classifier_server.py
```

Start `fuzzy_allocator_server.py` in a second terminal using the same
environment. The Java client expects both services at `127.0.0.1`.

Example requests:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/predict_priority `
  -Method Post -ContentType "application/json" `
  -Body '{"traffic_density":0.8,"time_of_day":1,"weather":2,"visibility":0.2}'

Invoke-RestMethod http://127.0.0.1:8080/allocate_resources `
  -Method Post -ContentType "application/json" `
  -Body '{"priority":"HIGH"}'
```

### 2. Run an iFogSim example

Open [`iFogSim/`](./iFogSim/) in IntelliJ IDEA, add the JARs in
[`iFogSim/jars/`](./iFogSim/jars/) as project libraries, and run a Java class
with a `main` method, such as
`org.fog.test.perfeval.DCNSFog` or `iFogSimulator.VehicleSimulation`.
The included [`iFogSim/README.md`](./iFogSim/README.md) contains the original
iFogSim setup notes and references.

## Notes

- Use Java 8 for the IntelliJ module configuration.
- Run the Python services before any simulation that calls `PythonClient`.
- Generated simulation output is written under `iFogSim/output/` and reports
  are stored under `iFogSim/results/`.
