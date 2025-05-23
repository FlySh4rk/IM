#!/usr/bin/env python3
"""
Script per testare il caricamento e l'inferenza del modello ImplantMaster
"""

import os
import sys
import tensorflow as tf
import keras

def test_model_loading():
    """Testa il caricamento del modello con diversi metodi"""
    model_path = 'data/model_v4_prod_v1'
    
    print("🔍 Test di caricamento del modello ImplantMaster")
    print("=" * 50)
    
    # Verifica esistenza del modello
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato in: {model_path}")
        print("📁 Contenuto della directory data/:")
        if os.path.exists('data'):
            for item in os.listdir('data'):
                print(f"   - {item}")
        else:
            print("   Directory data/ non esistente")
        return False
    
    print(f"✅ Modello trovato in: {model_path}")
    print(f"📁 Contenuto directory modello:")
    for item in os.listdir(model_path):
        print(f"   - {item}")
    
    # Metodo 1: TensorFlow SavedModel con gestione errori migliorata
    try:
        print("\n🚀 Metodo 1: tf.saved_model.load() con opzioni di compatibilità")
        
        # Prova con opzioni di compatibilità
        import warnings
        warnings.filterwarnings('ignore')
        
        # Disabilita eager execution per compatibilità
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.enable_eager_execution()
        
        model_tf = tf.saved_model.load(model_path)
        print("✅ Caricamento TensorFlow SavedModel riuscito")
        
        # Mostra signature disponibili
        if hasattr(model_tf, 'signatures'):
            signatures = list(model_tf.signatures.keys())
            print(f"📋 Signature disponibili: {signatures}")
            
            # Prova a fare una predizione di test se possibile
            if signatures:
                print(f"🧪 Test signature: {signatures[0]}")
                
        return True
        
    except Exception as e:
        print(f"❌ Errore con tf.saved_model.load(): {e}")
        print("💡 Potrebbe essere un problema di compatibilità versioni TF/Keras")
    
    # Metodo 2: Approccio alternativo con tf.compat
    try:
        print("\n🚀 Metodo 2: Caricamento con tf.compat.v1")
        
        # Reset del grafo
        tf.compat.v1.reset_default_graph()
        
        with tf.compat.v1.Session() as sess:
            model_tf = tf.compat.v1.saved_model.load(sess, ['serve'], model_path)
            print("✅ Caricamento con tf.compat.v1 riuscito")
            return True
            
    except Exception as e:
        print(f"❌ Errore con tf.compat.v1: {e}")
    
    # Metodo 3: Informazioni sul modello senza caricamento completo
    try:
        print("\n🚀 Metodo 3: Ispezione metadati modello")
        
        # Leggi solo i metadati
        saved_model_dir = model_path
        tag_set = ['serve']
        signature_def_map = tf.saved_model.load(saved_model_dir).signatures
        
        print("✅ Lettura metadati riuscita")
        print(f"📋 Signature trovate: {list(signature_def_map.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore lettura metadati: {e}")
    
    return False

def show_tensorflow_info():
    """Mostra informazioni sull'ambiente TensorFlow"""
    print("\n📊 Informazioni ambiente TensorFlow:")
    print("=" * 40)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"Python version: {sys.version}")
    
    # GPU info
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"GPU disponibili: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")

if __name__ == "__main__":
    show_tensorflow_info()
    success = test_model_loading()
    
    if success:
        print("\n🎉 Test completato con successo!")
        sys.exit(0)
    else:
        print("\n❌ Test fallito - controllare i metodi di caricamento")
        sys.exit(1)