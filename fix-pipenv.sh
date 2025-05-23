#!/bin/bash

# Script per risolvere problemi con Pipfile e tensorflow-metal

echo "🔧 Fix dei problemi con Pipfile e tensorflow-metal..."
echo "===================================================="

# Rimuovi Pipfile.lock obsoleto
if [ -f "Pipfile.lock" ]; then
    echo "🗑️  Rimozione Pipfile.lock obsoleto..."
    rm -f Pipfile.lock
    echo "✅ Pipfile.lock rimosso"
else
    echo "ℹ️  Pipfile.lock non presente"
fi

# Fix del Pipfile per rimuovere tensorflow-metal
if [ -f "Pipfile" ]; then
    echo "🔧 Fix del Pipfile per compatibilità Linux..."
    # Backup del Pipfile originale
    cp Pipfile Pipfile.backup
    # Rimuovi tensorflow-metal che è specifico per macOS
    sed -i '/tensorflow-metal/d' Pipfile
    # Assicurati che tensorflow standard sia presente
    if ! grep -q "tensorflow = " Pipfile; then
        # Aggiungi tensorflow nella sezione [packages]
        sed -i '/\[packages\]/a tensorflow = "*"' Pipfile
        echo "✅ tensorflow standard aggiunto al Pipfile"
    else
        echo "✅ tensorflow standard già presente nel Pipfile"
    fi
    echo "✅ tensorflow-metal rimosso dal Pipfile"
    echo "💾 Backup del Pipfile originale salvato come Pipfile.backup"
else
    echo "⚠️  Pipfile non trovato. Creazione di un Pipfile di base..."
    cat > Pipfile << 'EOF'
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
tensorflow = "*"
keras = "*"
numpy = "*"
pandas = "*"
matplotlib = "*"
seaborn = "*"
scikit-learn = "*"
jupyter = "*"
jupyterlab = "*"

[dev-packages]

[requires]
python_version = "3.9"
EOF
    echo "✅ Pipfile creato"
fi

# Pulisci immagini Docker esistenti
echo "🧹 Pulizia immagini Docker obsolete..."
docker compose down 2>/dev/null || true
docker system prune -f
docker builder prune -f

# Mostra il contenuto del Pipfile modificato
echo ""
echo "📋 Contenuto del Pipfile dopo il fix:"
echo "======================================"
cat Pipfile

# Ricostruisci le immagini
echo "🐳 Ricostruzione immagini Docker..."
docker compose build --no-cache

echo ""
echo "✅ Fix completato!"
echo ""
echo "🚀 Ora puoi avviare i servizi con:"
echo "   docker compose up -d"
echo "   # oppure"
echo "   make up"