#!/bin/bash

# Script di setup per il progetto ImplantMaster containerizzato

set -e

echo "🚀 Setup del progetto ImplantMaster"
echo "=================================="

# Verifica prerequisiti
echo "📋 Verifica prerequisiti..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker non è installato. Installa Docker prima di continuare."
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose non è installato. Installa Docker Compose prima di continuare."
    exit 1
fi

echo "✅ Docker e Docker Compose sono installati"

# Clona il repository se non esiste
if [ ! -d ".git" ]; then
    echo "📥 Clonazione del repository..."
    git clone https://github.com/FlySh4rk/IM.git .
else
    echo "✅ Repository già presente"
fi

# Crea le directory necessarie
echo "📁 Creazione directory..."
mkdir -p {data,models,validation,prod_data,test,src,backups}

# Verifica la presenza del Pipfile
echo "ℹ️  Uso del Pipfile esistente con fix per compatibilità Docker/Linux"

# Rimuovi Pipfile.lock obsoleto se presente
if [ -f "Pipfile.lock" ]; then
    echo "🧹 Rimozione Pipfile.lock obsoleto..."
    rm -f Pipfile.lock
fi

# Costruisci le immagini Docker
echo "🐳 Costruzione immagini Docker..."
docker compose build

# Avvia i servizi
echo "🚀 Avvio servizi..."
docker compose up -d

# Attendi che i servizi siano pronti
echo "⏳ Attesa avvio servizi..."
sleep 10

# Verifica lo stato
echo "📊 Stato servizi:"
docker compose ps

# Informazioni finali
echo ""
echo "✅ Setup completato!"
echo ""
echo "🎯 Comandi utili:"
echo "  make help           - Mostra tutti i comandi disponibili"
echo "  make jupyter        - Avvia Jupyter Lab (http://localhost:8888)"
echo "  make shell          - Accedi al container"
echo "  make test           - Esegui i test"
echo "  make logs           - Mostra i log"
echo ""
echo "📊 Per il training del modello:"
echo "  1. make jupyter"
echo "  2. Apri http://localhost:8888"
echo "  3. Naviga a models/model_v4_train.ipynb"
echo ""
echo "🔧 Per preprocessing dei dati:"
echo "  make preprocess"
echo ""
echo "📈 Per verificare il modello:"
echo "  make inference"

# Controlla se Jupyter è in esecuzione
if docker compose ps jupyter | grep -q "Up"; then
    echo ""
    echo "🎉 Jupyter Lab è disponibile su: http://localhost:8888"
fi