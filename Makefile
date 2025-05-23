# Makefile per gestire facilmente il progetto ImplantMaster

.PHONY: help build up down logs shell jupyter test train clean fix-pipenv

# Default target
help:
	@echo "Comandi disponibili per ImplantMaster:"
	@echo "  build      - Costruisce le immagini Docker"
	@echo "  up         - Avvia i servizi"
	@echo "  down       - Ferma i servizi"
	@echo "  logs       - Mostra i log"
	@echo "  shell      - Accede al container principale"
	@echo "  jupyter    - Avvia solo Jupyter Lab"
	@echo "  test       - Esegue i test"
	@echo "  train      - Esegue il training del modello"
	@echo "  preprocess - Preprocessa i dati"
	@echo "  clean-all  - Pulizia completa Docker (libera spazio)"
	@echo "  build-main - Costruisce solo container principale"
	@echo "  build-jupyter - Costruisce solo container Jupyter"

# Verifica contenuto del Pipfile
check-pipfile:
	@chmod +x check-pipfile.sh
	@./check-pipfile.sh
	@echo "  model-info - Mostra informazioni dettagliate sul modello"

# Costruisce le immagini Docker
build:
	docker compose build

# Avvia tutti i servizi
up:
	docker compose up -d

# Ferma i servizi
down:
	docker compose down

# Mostra i log
logs:
	docker compose logs -f

# Accede al container principale
shell:
	docker compose exec implantmaster bash

# Avvia solo Jupyter Lab
jupyter:
	docker compose up -d jupyter
	@echo "Jupyter Lab disponibile su: http://localhost:8888"

# Esegue i test
test:
	docker compose exec implantmaster python -m unittest discover test/

# Esegue il test specifico per la scrittura del dataset
preprocess:
	docker compose exec implantmaster python -m unittest dataset_tests.DataSetTestCase.test_write

# Esegue il training (richiede Jupyter)
train:
	@echo "Per il training, usa Jupyter Lab su http://localhost:8888"
	@echo "Apri il notebook: models/model_v4_train.ipynb"

# Esegue inferenza con il modello pre-addestrato (metodo compatibile)
inference:
	@echo "🔍 Verifica stato container..."
	@docker compose ps
	@echo "🚀 Caricamento modello con tf.compat.v1 (metodo funzionante)..."
	docker compose exec implantmaster python quick_inference.py

# Alternativa con TFSMLayer per Keras 3
inference-keras:
	@echo "🚀 Caricamento modello con Keras 3 TFSMLayer..."
	docker compose exec implantmaster python -c "import keras; layer = keras.layers.TFSMLayer('data/model_v4_prod_v1', call_endpoint='serving_default'); print('Modello caricato come TFSMLayer con successo')"

# Versione alternativa con script Python separato
# Test semplice esistenza modello
test-model:
	@echo "🧪 Test disponibilità del modello..."
	docker compose exec implantmaster python -c "import os; print('Modello presente:', os.path.exists('data/model_v4_prod_v1')); print('Contenuto:', os.listdir('data/model_v4_prod_v1') if os.path.exists('data/model_v4_prod_v1') else 'N/A')"

# Test TensorFlow semplice
test-tf:
	@echo "🔧 Test TensorFlow..."
	docker compose exec implantmaster python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('✅ TensorFlow funziona')"

# Script di inferenza pratico che funziona
run-inference:
	@echo "🎯 Esecuzione script di inferenza pratico..."
	docker compose exec implantmaster python working_inference.py
	@echo "  run-inference - Esegue inferenza con script pratico"

# Script completo per test modello
	@echo "🧪 Test completo di inferenza..."
	docker compose exec implantmaster python inference.py
	docker compose exec implantmaster python inference.py
# Mostra informazioni dettagliate sul modello
	@echo "📋 Informazioni dettagliate sul modello..."
	docker compose exec implantmaster python -c "import tensorflow as tf; import os; model_path='data/model_v4_prod_v1'; print('Modello esistente:', os.path.exists(model_path)); print('Contenuto directory modello:', os.listdir(model_path) if os.path.exists(model_path) else 'N/A'); model = tf.saved_model.load(model_path); print('Signature keys:', list(model.signatures.keys()) if hasattr(model, 'signatures') else 'N/A')"

# Script completo per test modello
test-inference:
	@echo "🧪 Test completo di inferenza..."
	docker compose exec implantmaster python inference.py
	docker compose exec implantmaster python -c "import tensorflow as tf; import os; model_path='data/model_v4_prod_v1'; print('Modello esistente:', os.path.exists(model_path)); print('Contenuto directory modello:', os.listdir(model_path) if os.path.exists(model_path) else 'N/A'); model = tf.saved_model.load(model_path); print('Signature keys:', list(model.signatures.keys()) if hasattr(model, 'signatures') else 'N/A')"

# Fix per problemi di compatibilità TensorFlow/Keras
fix-tensorflow:
	@echo "🔧 Fix compatibilità TensorFlow/Keras..."
	@echo "Creazione requirements.txt con versioni compatibili..."
	cp requirements-fixed.txt requirements.txt
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@echo "✅ Fix completato - versioni TensorFlow/Keras compatibili installate"

# Risolve problemi con container che si riavvia
fix-restart:
	@echo "🔧 Fix problema restart container..."
	docker compose down
	docker compose build --no-cache
	docker compose up -d
	@echo "✅ Container riparato"

# Debug: mostra i log del container principale
debug:
	@echo "🐛 Log del container implantmaster:"
	docker compose logs --tail=50 implantmaster

# Risolve problemi con tensorflow-metal nel Pipfile
fix-pipenv:
	@echo "🔧 Fix problemi tensorflow-metal..."
	rm -f Pipfile.lock
	./fix-pipenv.sh
	docker compose build --no-cache
	@echo "✅ Pipfile riparato, ricostruisci con: make build"

# Pulizia completa Docker per liberare spazio
clean-all:
	@echo "🧹 Pulizia completa Docker per liberare spazio..."
	docker compose down -v
	docker system prune -af
	docker volume prune -f
	docker builder prune -af
	docker image prune -af
	@echo "✅ Spazio liberato - controlla con: docker system df"

# Costruisce solo il container principale (più leggero)
build-main:
	@echo "🏗️  Build solo container principale..."
	docker compose build implantmaster

# Costruisce solo Jupyter (versione leggera)
build-jupyter:
	@echo "🏗️  Build solo container Jupyter..."
	docker compose build jupyter

# Pulisce container e volumi
clean:
	docker compose down -v
	docker system prune -f
	docker volume prune -f

# Backup del modello addestrato
backup-model:
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	docker compose exec implantmaster tar -czf /tmp/model_backup_$$timestamp.tar.gz data/model_v4_prod_v1; \
	docker cp $$(docker compose ps -q implantmaster):/tmp/model_backup_$$timestamp.tar.gz ./backups/; \
	echo "Backup salvato in: ./backups/model_backup_$$timestamp.tar.gz"

# Ripristina un backup del modello
restore-model:
	@read -p "Inserisci il nome del file di backup: " backup_file; \
	docker cp ./backups/$$backup_file $$(docker compose ps -q implantmaster):/tmp/; \
	docker compose exec implantmaster tar -xzf /tmp/$$backup_file -C /; \
	echo "Modello ripristinato da: $$backup_file"

# Mostra lo stato dei servizi
status:
	docker compose ps

# Ricostruisce e riavvia tutto
rebuild:
	docker compose down
	docker compose build --no-cache
	docker compose up -d