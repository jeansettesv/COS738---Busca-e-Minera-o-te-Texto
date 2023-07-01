#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import string
import csv
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom as dom
import math
import operator
import logging
import configparser


# In[ ]:


def numLinhas(arquivos):
    '''
    Verifica a quantidade de linhas de um arquivo.
    '''    
    contador_linhas = 0 
    try:
        for arquivo in arquivos:
            current_directory = os.getcwd()
            parent_directory  = os.path.dirname(current_directory)
            path = os.path.join(parent_directory,'data',arquivo)

            with open(path, 'r') as arquivo:
                for linha in arquivo:
                    contador_linhas += 1
        
    except:
        try:
            for arquivo in arquivos:
                current_directory = os.getcwd()
                parent_directory  = os.path.dirname(current_directory)
                path = os.path.join(parent_directory,'data',arquivo)
                
                with open(path, 'r') as arquivo:
                    for linha in arquivo:
                        contador_linhas += 1
        except:
            for arquivo in arquivos:
                current_directory = os.getcwd()
                parent_directory  = os.path.dirname(current_directory)
                path = os.path.join(parent_directory,'result',arquivo)
                
                with open(path, 'r') as arquivo:
                    for linha in arquivo:
                        contador_linhas += 1
                    
    return contador_linhas
    

def ler_tags(tags, arquivo):
    '''
    Lê as ocorrências das Tags em um arquivo XML.
    '''        
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'data',arquivo)
    
   # Leitura do arquivo XML
    doc = dom.parse(path)

    # Creiacao da Matrix onde as colunas são as tags e as linhas são os registros da tag pai
    result = []
    for tag in tags:
        result.append([elemento.firstChild.data for elemento in doc.getElementsByTagName(tag)])
    return [tags] + np.matrix(result).T.tolist()
    
    
def ler_tags_case(tags, arquivos):
    '''
    Lê as ocorrências das Tag em um arquivo XML e caso uma das tag for ABSTRACT e a entidade corrente não tiver, lerá a EXTRACT.
    ''' 
    result_final = []
    for arquivo in arquivos:
        current_directory = os.getcwd()
        parent_directory  = os.path.dirname(current_directory)
        path = os.path.join(parent_directory,'data',arquivo)

        # Leitura do arquivo XML
        doc  = dom.parse(path)
        tree = ET.parse(path)  
        root = tree.getroot()

        result = []    
        for tag in tags:
            
            if tag == 'ABSTRACT': 
                
                values = []
                for record in root.findall('RECORD'):
                    primeira_tag = record.find('ABSTRACT')
                    segunda_tag = record.find('EXTRACT')
                    
                    if primeira_tag is not None:
                        primeira_tag_valor = primeira_tag.text
                        values.append(primeira_tag_valor)
                        
                    elif segunda_tag is not None:
                        segunda_tag_valor = segunda_tag.text
                        values.append(segunda_tag_valor)
                    else:
                        values.append('')
                        
                result.append(values)
                
            else:
                result.append([elemento.firstChild.data.replace(' ','') for elemento in doc.getElementsByTagName(tag)])
                
        result_final = result_final + list([list(row) for row in zip(*result)])

    return [tags] + result_final


def ler_tag_attr(tag, attr, arquivo):
    '''
    Lê o atributo de uma tag em um arquivo XML.
    '''  
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'data',arquivo)
    
    # Leitura do arquivo XML
    doc = dom.parse(path)

    # Criacao da Matrix onde as colunas são as tags e as linhas são os registros da tag pai
    result = []
    elementos = doc.getElementsByTagName(tag)
    for element in elementos:
        result.append([element.firstChild.data, element.getAttribute(attr)])
    
    return [[tag, attr]] + result    


def CriaConsulta(texto):
    '''
    Cria uma consulta baseado nas palavras chaves do texto.
    '''    
    texto_sem_acentos = unidecode(texto)                                                             
    texto_maiusculo = texto_sem_acentos.upper()                                                      
    texto_sem_quebra_e_aspas = texto_maiusculo.replace('\n',' ').replace('"','').replace('-',' ')    
    
    tokens = word_tokenize(texto_sem_quebra_e_aspas)
    pontuacoes = set(string.punctuation)
    
    palavras = [token for token in tokens if token not in pontuacoes and token.lower() not in stopwords.words('english')] 
    texto = ' '.join(palavras)
    
    return texto
    
    
def processa_arq_consulta(tags_originais):
    '''
    Gera a lista que será gravada no arquivo citado na instrução CONSULTA do arquivo PC.CFG.
    '''
    tags_tratadas = [[tags_originais[0][0], tags_originais[0][1]]] + [[i[0],CriaConsulta(i[1])] for i in tags_originais[1:]]
    
    return tags_tratadas
        
    
def escreve_arq_consulta(tags_tratadas, nome_arq_saida):
    '''
    A partir do resultadoda processa_arq_consulta, escreve o arquivo que é o citado na instrução CONSULTA do arquivo PC.CFG.
    '''    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_saida)
    
    with open(path, 'w', newline='') as arquivo_csv:
        writer = csv.writer(arquivo_csv, delimiter=';')
        writer.writerows(tags_tratadas)
        

def processa_arq_esperados(resultado1, resultado2):
    '''
    Gera a lista contendo o numero de votos de cada documento recuperado pela consulta, o numero do documento e o numero 
    da consulta. Esta lista será gravada no arquivo citado na instrução ESPERADOS do arquivo PC.CFG.
    '''
    header = ['QueryNumber', 'DocNumber', 'DocVotes']
    result1 = resultado1[1:]
    result2 = resultado2[1:]
    
    avalIndex = 0
    result = []
    for i in result1:
        
        queryNum = i[0]
        avaliacoes = int(i[1])
        for j in range(avaliacoes):
            
            document = result2[avalIndex][0]
            aval = np.sum([1 for k in result2[avalIndex][1] if int(k)>0])
            
            linha = [queryNum, str(document).zfill(5), aval] 
            result.append(linha)
            avalIndex = avalIndex + 1
    
    return [header] + result


def escreve_arq_esperados(arq_esperados, nome_arq_saida):
    '''
    A partir do resultado da processa_arq_esperados, escreve o arquivo que é o citado na instrução ESPERADOS do arquivo PC.CFG.
    '''        
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_saida)
    
    with open(path, 'w', newline='') as arquivo_csv:
        writer = csv.writer(arquivo_csv, delimiter=';')
        writer.writerows(arq_esperados)

        
def processa_arq_lista_invertida(lista):    
    '''
    Cria uma lista_invertida para as palavras encontradas nos ABSTRAC/ EXTRACT dos documentos contidos nos arquivos de entrada.
    '''    
    lista_invertida = {}
    for linha in lista[1:]:
        
        numero_documento = linha[0]
        texto = linha[1]
        
        texto_sem_acentos = unidecode(texto)                                                            
        texto_maiusculo = texto_sem_acentos.upper()                                                     
        texto_sem_quebra_e_aspas = texto_maiusculo.replace('\n',' ').replace('"','').replace('-',' ')

        tokens = word_tokenize(texto_sem_quebra_e_aspas)
        pontuacoes = set(string.punctuation)
    
        palavras = [token for token in tokens if token not in pontuacoes and token.lower() not in stopwords.words('english')]
        for palavra in palavras:
            
            if palavra not in lista_invertida:
                lista_invertida[palavra] = []
                
            lista_invertida[palavra].append(numero_documento)
    
    return lista_invertida


def escreve_arq_lista_invertida(lista_invertida, nome_arq_saida):
    '''
    Escreve a lista invertida no qual guarda os documentos em que cada termo aparece (mantendo repetições) em um arquivo cujo 
    nome é citado na instrução ESCREVA do arquivo GLI.CFG.
    '''        
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_saida)
    
    with open(path, 'w', newline='') as file:

        writer = csv.writer(file, delimiter=';')
        writer.writerow(['PALAVRA', 'DOCUMENTOS'])
        for palavra, documentos in lista_invertida.items():
            
            writer.writerow([palavra, documentos])


def processa_arq_index(nome_arq_listaI, norm=0):
    '''
    Cria uma matrix termo-documento na forma de um dicionário onde para cada documento temos os termos que aparecem (e não 
    aparecem) neles e o TF-IDF que qualifica a relevância do termo no respectivo documento. Vale ressaltar que como resolvi 
    guardar a informação da base da matriz para facilitar cálculos futuros, para cada documento, então, foi adicionado também 
    uma entrada para os termos que não aparecem nele mas está presente em algum dos demais. Além disso, foi dado o valor 0
    para seu TF-IDF.
    '''        
    dic_freq_docs = {}
    dic_freq_words = {}
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_listaI)

    with open(path, 'r') as arquivo:
        linhas = arquivo.readlines()[1:]

        for linha in linhas:
            word, docs = linha.strip().split(';')
            doc_list = [doc.strip().replace('[','').replace(']','').replace("'", "") for doc in docs.split(',')]  # Converte a string em uma 
                                                                                                                  # lista de documentos
                            
            if word.isalpha() and len(word) >= 2:
            
                for doc in doc_list:
                    
                    if doc not in dic_freq_docs:
                        dic_freq_docs[doc] = {}
                    
                    if word not in dic_freq_docs[doc]:
                        dic_freq_docs[doc][word] = 0
                        
                    dic_freq_docs[doc][word] += 1
                
                if word not in dic_freq_words:
                    dic_freq_words[word] = 0
                    
                dic_freq_words[word] += len(set(doc_list)) # Adiciona no dicionário o total de documentos em que a palavra aparece
                
    # Cálculo do IDF
    total_docs = len(dic_freq_docs)
    dic_idf = {word: math.log(total_docs / freq) for word, freq in dic_freq_words.items()}

    # Cálculo do TF-IDF normalizado
    index_tf_idf = {}
    for doc, word_freqs in dic_freq_docs.items():
        index_tf_idf[doc] = {}
        
        for word, freq in dic_freq_words.items():
            
            if word in word_freqs:
                index_tf_idf[doc][word] = (1 + math.log(freq)) * dic_idf[word] if norm != 0 else 1 * dic_idf[word]
                
    return index_tf_idf


def escreve_arq_index(index, nome_arq_saida):
    '''
    A partir do resultado da processa_arq_index, escreve o arquivo que é o citado na instrução ESPERADOS do arquivo INDEX.CFG.
    ''' 
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_saida)
    
    with open(path, 'w', newline='') as arquivo:
        writer = csv.writer(arquivo, delimiter=';')
        writer.writerow(["Documento", "Palavra", "TF-IDF"])

        for doc, word_freqs in index.items():
            for word, tf_idf in word_freqs.items():
                writer.writerow([doc, word, tf_idf])
                
                
def carrega_modelo_vetorial(nome_arq_modelo):
    '''
    Lê o arquivo que contém o modelo vetorial e transforma em um dicionário contendo os documentos e os pesos das palavras da
    base matricial na sua respectiva representação vetorial (tf-idf ou 0).
    '''
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_modelo)
    
    index_tf_idf = {}
    with open(path, 'r') as arquivo:
        reader = csv.reader(arquivo, delimiter=';')
        next(reader)  

        for linha in reader:
            doc, word, tf_idf = linha
            if doc not in index_tf_idf:
                index_tf_idf[doc] = {}
            index_tf_idf[doc][word] = float(tf_idf)

    return index_tf_idf


def calcular_vetor_consulta(consulta, vetor_doc):
    '''
    A consulta aqui é uma das consultas do arquivo que contem as consultas, e a partir dela geramos um vetor de comprimento 
    igual à base de palavras do nosso modelo vetorial com o auxílio de um dos vetores documentos presente no modelo. Atrubuímos
    o peso 1 para as palavras em comum e 0 para as palavras da base que não pertencem à consulta. Retorna o vetor gerado.
    '''
    palavras = list(vetor_doc.keys())  
    total_palavras = len(palavras)
    
    vetor_consulta = np.zeros(total_palavras)
    for palavra, freq in consulta.items():
        
        if palavra in palavras:
            indice = palavras.index(palavra)
            vetor_consulta[indice] = 1
            
    return vetor_consulta



def calcular_similaridade(vetor_consulta, vetor_doc):
    '''
    Calacula a similaridade entre o vetor_consulta e o vetor_doc utilizando o cosseno entre eles. 
    '''    
    vetor_consulta = np.array(vetor_consulta)
    vetor_doc = np.array(list(vetor_doc.values()))
    
    if np.linalg.norm(vetor_consulta) == 0 or np.linalg.norm(vetor_doc) == 0:
        similaridade = 0
    else:
        similaridade = np.dot(vetor_consulta, vetor_doc) / (np.linalg.norm(vetor_consulta) * np.linalg.norm(vetor_doc))
    
    return similaridade


def processa_arq_busca(nome_arq_modelo, nome_arq_consultas):
    '''
    Realiza a busca dos documentos mais relevantes para cada consulta de acordo com a similaridade entre eles. O retorno dessa
    funcao é uma lista contendo as consultas e uma lista de tuplas que contem a posição do ranking de similaridade, os documentos 
    e os valores de similaridade, isso para cada consulta. O nome do arquivo de consultas se encontra na sessão CONSULTAS do 
    arquivo BUSCA.CGF e o nome do arquivo modelo se encontra na sessão MODELO do mesmo arquivo.
    '''
    modelo = carrega_modelo_vetorial(nome_arq_modelo)
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arq_consultas)
    
    consultas = {}
    with open(path, 'r') as arquivo:
        reader = csv.reader(arquivo, delimiter=';')
        next(reader)  
        
        for linha in reader:
            numquery, query = linha
            consultas[numquery] = {}

            # Processar a consulta para extrair as palavras e ocorrências
            palavras = query.split()  # Separa as palavras da consulta
            for palavra in palavras:
                
                if palavra.isalpha() and len(palavra) >= 2:
                    
                    if palavra not in consultas[numquery]:
                        consultas[numquery][palavra] = 1
                
    resultados = []
    for consulta_id, consulta in consultas.items():
                
        # Calcular a similaridade entre a consulta e os documentos
        similaridade = {}
        for doc, vetor_doc in modelo.items():
            vetor_consulta = calcular_vetor_consulta(consulta, vetor_doc)
            similaridade[doc] = calcular_similaridade(vetor_consulta, vetor_doc)

        # Ordenar os documentos por similaridade
        documentos_ordenados = sorted(similaridade.items(), key=operator.itemgetter(1), reverse=True)
        
        # Criar a lista de ternos ordenados
        lista_ternos = [(posicao, doc, distancia) for posicao, (doc, distancia) in enumerate(documentos_ordenados, start=1)]

        # Adicionar o resultado à lista de resultados
        resultados.append((consulta_id, lista_ternos))
    
    return resultados
    
    
def escreve_arq_busca(busca, nome_arquivo):
    '''
    Escreve o resultado da função processa_arq_busca em um arquivo cujo nome se encontra na sessão RESULTADOS do arquivo 
    BUSCA.CFG.
    '''
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    path = os.path.join(parent_directory,'result',nome_arquivo)
    
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Consulta', 'Resultado'])

        for query_id, query_results in busca:
            ranked_results = sorted(query_results, key=lambda x: x[0])
            result_str = str([(rank, doc_id, distance) for rank, doc_id, distance in ranked_results])
            writer.writerow([query_id, result_str.replace('"','').replace("'",'')])


# In[ ]:


# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Criar os manipuladores de console e arquivo e atribuir o Formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
path = os.path.join(parent_directory, 'log.txt')

file_handler = logging.FileHandler(path, 'w')
file_handler.setFormatter(formatter)

# Configurar o logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Adicionar os manipuladores ao logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Limpando a saída
logger.handlers.clear()


def ler_dados():
    resultado = {}
    
    # Ler arquivo de configuração
    logger.info('Iniciando leitura do arquivos de configuração...')
    
    current_directory = os.getcwd()
    parent_directory  = os.path.dirname(current_directory)
    config            = configparser.ConfigParser()
    linhas_config     = 0
    
    arquivo_config              = os.path.join(parent_directory, 'PC.cfg')
    config.read(arquivo_config)    
    resultado['leia_pc_r']      = config.get('LEIA', 'nome_arquivo')
    resultado['concultas_pc_w'] = config.get('CONSULTAS', 'nome_arquivo')
    resultado['esperados_pc_w'] = config.get('ESPERADOS', 'nome_arquivo')
    linhas_config              += 4

    arquivo_config             = os.path.join(parent_directory, 'GLI.cfg')
    config.read(arquivo_config) 
    resultado['leia_gli_r']    = [value for key, value in config.items('LEIA')]
    resultado['escreva_gli_w'] = config.get('ESCREVA', 'nome_arquivo')
    linhas_config             += 2+len(resultado['leia_gli_r'])
    
    arquivo_config               = os.path.join(parent_directory, 'INDEX.cfg')
    config.read(arquivo_config) 
    resultado['leia_index_r']    = config.get('LEIA', 'nome_arquivo')
    resultado['escreva_index_w'] = config.get('ESCREVA', 'nome_arquivo')
    linhas_config               += 3
    
    arquivo_config                  = os.path.join(parent_directory, 'BUSCA.cfg')
    config.read(arquivo_config) 
    resultado['modelo_busca_r']     = config.get('MODELO', 'nome_arquivo')
    resultado['consultas_busca_r']  = config.get('CONSULTAS', 'nome_arquivo')
    resultado['resultados_busca_w'] = config.get('RESULTADOS', 'nome_arquivo')
    linhas_config                  += 4
    
    logger.info('Arquivos de configuração lidos.\n')

    # Ler arquivo de dados
    logger.info('Iniciando leitura do arquivo de dados...')
    
    resultado['tags1']    = ['QueryNumber', 'QueryText']
    resultado['tags2']    = ['QueryNumber', 'Results']
    resultado['tags3']    = ['RECORDNUM', 'ABSTRACT']
    resultado['tag_attr'] = 'Item'
    resultado['attr']     = 'score'
    
    linhas_dados = numLinhas([resultado['leia_pc_r']]) + numLinhas(resultado['leia_gli_r'])                 + numLinhas([resultado['leia_index_r']]) + numLinhas([resultado['modelo_busca_r']])                 + numLinhas([resultado['modelo_busca_r']])
    
    logger.info(f'Arquivo de dados lidos.')
    logger.info(f'Total de linhas lidas: {linhas_config + linhas_dados}.\n\n\n')
    
    return resultado

def processar_dados(resultado):
    resultado2 = {}
    
    # Processamento de consultas
    logger.info('Iniciando processamento de consultas...')
    inicio = time.time()
    
    result_leia1_pc = ler_tags(resultado['tags1'], resultado['leia_pc_r'])
    result_leia2_pc = ler_tags(resultado['tags2'], resultado['leia_pc_r'])
    result_leia3_pc = ler_tag_attr(resultado['tag_attr'], resultado['attr'], resultado['leia_pc_r'])
    
    resultado2['arq_consulta']  = processa_arq_consulta(result_leia1_pc)
    resultado2['arq_esperados'] = processa_arq_esperados(result_leia2_pc, result_leia3_pc)
    
    fim = time.time()
    tempo_medio_consultas = fim - inicio
    logger.info(f'Consultas Processadas. Tempo médio de processamento de consultas: {tempo_medio_consultas} segundos\n')

    # Processamento de documentos
    logger.info('Iniciando processamento dos documentos...')
    inicio = time.time()
        
    result_leia_gli = ler_tags_case(resultado['tags3'], resultado['leia_gli_r'])
    resultado2['lista_invertida'] = processa_arq_lista_invertida(result_leia_gli)
    
    fim = time.time()
    tempo_medio_documentos = fim - inicio
    logger.info(f'Documentos processados. Tempo médio de processamento de documentos: {tempo_medio_documentos} segundos\n')

    # Processamento de palavras
    logger.info('Iniciando processamento de palavras...')
    inicio = time.time()
    resultado2['result_leia_index'] = processa_arq_index(resultado['leia_index_r'])
    resultado2['result_busca'] = processa_arq_busca(resultado['modelo_busca_r'], resultado['consultas_busca_r'])
    
    fim = time.time()
    tempo_medio_palavras = fim - inicio
    logger.info(f'Palavras processadas. Tempo médio de processamento de palavras: {tempo_medio_palavras} segundos\n\n\n')
    
    return resultado2
    

def salvar_dados(resultado, resultado2):
    
    logger.info('Iniciando salvamento...')
    escreve_arq_consulta(resultado2['arq_consulta'], resultado['concultas_pc_w'])
    escreve_arq_esperados(resultado2['arq_esperados'], resultado['esperados_pc_w'])
    escreve_arq_lista_invertida(resultado2['lista_invertida'], resultado['escreva_gli_w'])
    escreve_arq_index(resultado2['result_leia_index'], resultado['escreva_index_w'])
    escreve_arq_busca(resultado2['result_busca'], resultado['resultados_busca_w'])
    
    logger.info('Dados salvos\n\n\n')

