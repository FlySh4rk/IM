# ImplantMaster - Containerizzazione Docker

Questa guida ti aiuterà a containerizzare e utilizzare il progetto ImplantMaster utilizzando Docker e Docker Compose.

## 🚀 Quick Start

### Prerequisiti
- Docker (versione 20.10+)
- Docker Compose (versione 1.29+)
- Git

**Nota**: Il progetto mantiene l'uso di `Pipfile` ma viene automaticamente corretto per rimuovere `tensorflow-metal` (specifico per macOS) e renderlo compatibile con Docker/Linux.

### Setup Automatico
```bash
# Clona il repository
git clone https://github.com/FlySh4rk/IM.git
cd IM

# Copia i file Docker nella directory del progetto
# (dopo aver scaricato i file dalla conversazione)

# Esegui il setup automatico
chmod +x setup.sh
./setup.sh
```

### Setup Manuale
```bash
# Costruisci le immagini
docker compose build

# Avvia i servizi
docker compose up -d

# Verifica lo stato
docker compose ps
```

## 🐳 Architettura dei Container

### Servizi Disponibili

1. **implantmaster** - Container principale con l'applicazione
2. **jupyter** - Jupyter Lab per sviluppo e training

### Volumi Persistenti
- `models_data` - Modelli addestrati
- `jupyter_data` - Configurazioni Jupyter

## 📋 Comandi Principali

### Utilizzo con Makefile
```bash
# Mostra tutti i comandi disponibili
make help

# Costruisci le immagini
make build

# Avvia i servizi
make up

# Accedi al container
make shell

# Avvia Jupyter Lab
make jupyter

# Esegui i test
make test

# Preprocessa i dati
make preprocess

# Ferma i servizi
make down
```

### Comandi Docker Compose Diretti
```bash
# Avvia tutti i servizi
docker compose up -d

# Mostra i log
docker compose logs -f

# Accedi al container principale
docker compose exec implantmaster bash

# Ferma i servizi
docker compose down

# Ricostruisci tutto
docker compose build --no-cache
```

## 🔬 Workflow di Sviluppo

### 1. Preprocessing dei Dati
```bash
# Preprocess dei dati di produzione
make preprocess

# Oppure manualmente
docker compose exec implantmaster python -m unittest dataset_tests.DataSetTestCase.test_write
```

### 2. Training del Modello
```bash
# Avvia Jupyter Lab
make jupyter

# Apri http://localhost:8888
# Naviga a models/model_v4_train.ipynb
# Esegui il notebook per il training
```

### 3. Testing e Validazione
```bash
# Esegui tutti i test
make test

# Test specifici
docker compose exec implantmaster python -m unittest dataset_tests.DataSetTestCase.test_read
```

### 4. Inferenza
```bash
# Verifica che il modello sia caricabile
make inference

# Oppure usa i notebook di validazione
# Apri validation/ in Jupyter Lab
```

## 📁 Struttura dei File

```
IM/
├── Dockerfile                 # Container principale
├── Dockerfile.jupyter        # Container Jupyter
├── docker compose.yml        # Configurazione servizi
├── .dockerignore             # File da escludere
├── Makefile                  # Comandi automatizzati
├── setup.sh                 # Script di setup
├── jupyter_notebook_config.py # Config Jupyter
├── backups/                  # Backup modelli
├── data/                     # Dati e modelli
├── models/                   # Notebook di training
├── prod_data/                # Dati di produzione
├── src/                      # Codice sorgente
├── test/                     # Test
└── validation/               # Notebook di validazione
```

## 🔧 Configurazione

### Variabili d'Ambiente
Il file `docker compose.yml` include le seguenti configurazioni:

- `PYTHONPATH=/app` - Percorso Python
- `TF_CPP_MIN_LOG_LEVEL=2` - Riduce log TensorFlow

### Personalizzazione
Per personalizzare la configurazione:

1. Modifica `docker compose.yml` per le variabili d'ambiente
2. Modifica `Dockerfile` per dipendenze aggiuntive
3. Modifica `jupyter_notebook_config.py` per configurazioni Jupyter

## 🛠️ Troubleshooting

### Problemi Comuni

**Errore di memoria durante il training:**
```bash
# Aumenta la memoria disponibile a Docker
# Settings → Resources → Memory
```

**Jupyter non accessibile:**
```bash
# Verifica che il servizio sia in esecuzione
docker compose ps

# Controlla i log
docker compose logs jupyter
```

**Errori di permessi:**
```bash
# Verifica proprietà dei file
docker compose exec implantmaster ls -la

# Ricrea i container se necessario
docker compose down -v
docker compose up -d
```

### Debug
```bash
# Accedi al container per debug
docker compose exec implantmaster bash

# Controlla l'ambiente Python
docker compose exec implantmaster pipenv --venv

# Verifica TensorFlow
docker compose exec implantmaster python -c "import tensorflow as tf; print(tf.__version__)"
```

## 🔒 Sicurezza

**⚠️ Nota Importante**: La configurazione Jupyter attuale è impostata per sviluppo locale senza autenticazione. Per ambienti di produzione:

**⚠️ Nota sulla Compatibilità macOS/Linux**: Se il `Pipfile` originale contiene `tensorflow-metal` (specifico per macOS con chip Apple Silicon), viene automaticamente rimosso durante il build Docker per garantire compatibilità con Linux. Il Pipfile originale viene salvato come backup in `Pipfile.backup`.

1. Abilita autenticazione in `jupyter_notebook_config.py`
2. Usa HTTPS
3. Configura firewall appropriato
4. Non esporre porte pubblicamente

## 📊 Monitoraggio

### Log dei Servizi
```bash
# Log di tutti i servizi
make logs

# Log specifici
docker compose logs implantmaster
docker compose logs jupyter
```

### Health Check
```bash
# Verifica stato dei container
docker compose ps

# Health check manuale
docker compose exec implantmaster python -c "import tensorflow as tf; print('OK')"
```

## 💾 Backup e Restore

### Backup del Modello
```bash
# Backup automatico
make backup-model

# Il backup sarà salvato in ./backups/
```

### Restore del Modello
```bash
# Restore da backup
make restore-model
# Inserisci il nome del file quando richiesto
```

## 🚀 Deployment

Per il deployment in produzione, considera:

1. **Multi-stage builds** per ottimizzare le dimensioni
2. **Docker secrets** per gestire credenziali
3. **Volume bind mounts** per dati persistenti
4. **Resource limits** per controllare l'uso di risorse
5. **Orchestration** con Kubernetes o Docker Swarm

## 📞 Supporto

Per problemi o domande:
1. Controlla i log: `make logs`
2. Verifica la configurazione: `docker compose config`
3. Ricrea l'ambiente: `make clean && make build && make up`