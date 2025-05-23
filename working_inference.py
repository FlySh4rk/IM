#!/usr/bin/env python3
"""
Script di inferenza funzionante per il modello ImplantMaster
Usa tf.compat.v1 che funziona con questo modello
"""

import os
import numpy as np
import tensorflow as tf

def load_model():
    """Carica il modello usando il metodo compatibile"""
    model_path = 'data/model_v4_prod_v1'
    
    print("🚀 Caricamento modello ImplantMaster...")
    
    # Reset del grafo
    tf.compat.v1.reset_default_graph()
    
    # Carica il modello con tf.compat.v1
    with tf.compat.v1.Session() as sess:
        model = tf.compat.v1.saved_model.load(sess, ['serve'], model_path)
        print("✅ Modello caricato con successo!")
        
        # Mostra informazioni sul modello
        signature_def = model.signature_def
        print(f"📋 Signature disponibili: {list(signature_def.keys())}")
        
        return model, sess

def example_inference():
    """Esempio di come usare il modello per inferenza"""
    print("\n🧪 Esempio di inferenza...")
    
    try:
        model, sess = load_model()
        print("✅ Modello pronto per inferenza!")
        
        # Qui puoi aggiungere il codice per fare predizioni
        # Ad esempio:
        # input_data = np.array([[1, 2, 3, 4]])  # I tuoi dati di input
        # prediction = sess.run(model.outputs, feed_dict={model.inputs: input_data})
        
        print("💡 Per fare predizioni, usa sess.run() con i tuoi dati di input")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante l'inferenza: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Script di inferenza ImplantMaster")
    print("=" * 40)
    
    # Verifica che il modello esista
    model_path = 'data/model_v4_prod_v1'
    if not os.path.exists(model_path):
        print(f"❌ Modello non trovato in: {model_path}")
        exit(1)
    
    # Test di caricamento
    success = example_inference()
    
    if success:
        print("\n🎉 Modello pronto per l'uso!")
        print("💡 Modifica questo script per aggiungere i tuoi dati di input")
    else:
        print("\n❌ Problemi con il caricamento del modello")
        exit(1)