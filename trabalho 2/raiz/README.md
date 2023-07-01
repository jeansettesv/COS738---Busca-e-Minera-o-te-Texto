# Trabalho 2 - Programa de busca

Este programa tem como objetivo buscar nos documentos disponibilizados, os documentos mais similares às consultas contidas no arquivo cfquery. Algumas informações importantes seguem adiante:

1) Ele esta separado como sugerido pelo professor, logo existe uma pasta src com os codigos fontes, uma pasta result com os resultados e a pasta raiz onde estão contidas as duas pastas e, além delas, os arquivos de configuração que são lidos, o log gerado, o README.md e o MODELO.txt.

2) O arquivo funciona da seguinte forma: Ele vai supor a existência dos arquivos listados nas configurações e, caso não existam, ele vai pedir pra verificá-los (isso vale para todos os arquivos com exceção do arquivo resultados.csv). 

3) Outro ponto relevante é que o programa primeiro ler todos os arquivos, logo ele vai produzir resultados de acordo com os arquivos pré-existentes, ou seja, não é algo sequencial (não utiliza o resultado das etapa anteriores), por isso ele não foi feito para alteração de arquivo, ele cria uma base com todos os documentos contido nos arquivos e faz a busca para todas as consultas. Se alterar algum arquivo, o programa precisará ser rodado mais de uma vez para alterar todos os arquivos uma vez que ele só escreve após o processamento de todos. O ideal seria uma opção mais versátil mas fiquei confuso com algumas orientações contidas na descrição do trabalho.

4) Não precisa dar imput, é só executar o arquivo main (seja o notebook ou a versão em .py).

No mais é isso, e qualquer dúvida entre em contato: jssv@cos.ufrj.br
