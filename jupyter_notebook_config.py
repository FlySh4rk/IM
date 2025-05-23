# Configurazione Jupyter Notebook
c = get_config()

# Permetti connessioni da qualsiasi IP
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888

# Disabilita autenticazione per sviluppo (NON usare in produzione)
c.NotebookApp.token = ''
c.NotebookApp.password = ''

# Consenti accesso root
c.NotebookApp.allow_root = True

# Non aprire browser automaticamente
c.NotebookApp.open_browser = False

# Directory di lavoro
c.NotebookApp.notebook_dir = '/app'

# Abilita estensioni
c.NotebookApp.nbserver_extensions = {
    'jupyterlab': True
}

# Configurazioni di sicurezza per sviluppo
c.NotebookApp.disable_check_xsrf = True