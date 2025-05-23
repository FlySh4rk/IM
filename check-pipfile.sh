#!/bin/bash

# Script per verificare il contenuto del Pipfile

echo "🔍 Verifica del Pipfile"
echo "======================="

if [ ! -f "Pipfile" ]; then
    echo "❌ Pipfile non trovato!"
    exit 1
fi

echo "📋 Contenuto attuale del Pipfile:"
echo "=================================="
cat Pipfile

echo ""
echo "🔍 Controllo dipendenze TensorFlow:"
echo "===================================="

# Verifica tensorflow-metal (dovrebbe essere assente)
if grep -q "tensorflow-metal" Pipfile; then
    echo "⚠️  tensorflow-metal trovato (problema per Linux)"
    grep "tensorflow-metal" Pipfile
else
    echo "✅ tensorflow-metal non presente (OK per Linux)"
fi

# Verifica tensorflow standard (dovrebbe essere presente)
if grep -q "tensorflow = " Pipfile; then
    echo "✅ tensorflow standard presente"
    grep "tensorflow = " Pipfile
else
    echo "❌ tensorflow standard NON presente - da aggiungere!"
fi

echo ""
echo "🏷️  Altre dipendenze TensorFlow/ML:"
echo "==================================="
grep -E "(keras|numpy|pandas|matplotlib|scikit-learn)" Pipfile || echo "Nessuna altra dipendenza ML trovata"

# Verifica se esiste backup
if [ -f "Pipfile.backup" ]; then
    echo ""
    echo "💾 Backup disponibile: Pipfile.backup"
    echo "Per ripristinare: mv Pipfile.backup Pipfile"
fi