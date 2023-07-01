#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Functions import *

logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Ler todos os dados
try:
    dados = ler_dados()
except:
    logger.info('Erro na leitura dos dados, favor verificar a integridade dos arquivos necessários.')

# Fazer todo o processamento
try:
    dados_processados = processar_dados(dados)
except:
    logger.info('Erro na leitura dos dados, favor verificar a integridade dos arquivos necessários.')


# Salvar todos os dados
try:
    salvar_dados(dados, dados_processados)
except:
    logger.info('Erro no salavamento dos dados, favor verificar a integridade dos arquivos necessários.')

logger.info('Programa finalizado')

