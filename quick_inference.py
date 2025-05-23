#!/usr/bin/env python3
"""
Script rapido per testare il caricamento del modello
"""

import tensorflow as tf

def main():
    try:
        print("🚀 Test caricamento modello con tf.compat.v1...")
        
        # Reset del grafo
        tf.compat.v1.reset_default_graph()
        
        # Carica il modello
        with tf.compat.v1.Session() as sess:
            model = tf.compat.v1.saved_model.load(sess, ['serve'], 'data/model_v4_prod_v1')
            print("✅ Modello caricato con successo usando tf.compat.v1")
            
            # Mostra signature se disponibili
            signature_def = model.signature_def
            if signature_def:
                print(f"📋 Signature disponibili: {list(signature_def.keys())}")
            
            return True
            
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)