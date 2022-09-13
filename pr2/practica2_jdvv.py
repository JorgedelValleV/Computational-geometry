"""
Práctica 2
"""

import os
import numpy as np
import pandas as pd
import math
#### Vamos al directorio de trabajo####
os.getcwd()
#os.chdir(ubica)
#files = os.listdir(ruta)

with open('GCOM2022_pract2_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2022_pract2_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


#### Contamos cuantas letras hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)
##### Transformamos en formato array de los carácteres (states) y su frecuencia
##### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))

##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}

## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
distr = distr_en 
tree = huffman_tree(distr)
tree[0].items()
tree[0].values()
#Buscar cada estado dentro de cada uno de los dos items
list(tree[0].items())[0][1] ## Esto proporciona un '0'
list(tree[0].items())[1][1] ## Esto proporciona un '1'

def calculate_codes(tree):
    codes=dict()
    for node in reversed(tree):##tre[::-1]
        for symbol in list(node.items())[0][0]:
            if codes.get(symbol) == None:  
                codes[symbol] =  '0'
            else:   
                codes[symbol] +=  '0'
        for symbol in list(node.items())[1][0]:
            if codes.get(symbol) == None:  
                codes[symbol] =  '1'
            else:   
                codes[symbol] +=  '1'
    return codes

def calculate_codes_df(tree,distr): # para estar en el mismo orden que el dataframe
    states= np.array(distr['states'])
    probab=np.array(distr['probab'])
    codes=dict.fromkeys(states,'')
    for node in reversed(tree):##tre[::-1]
        for symbol in list(node.keys())[0]:
            codes[symbol] +=  '0'
        for symbol in list(node.keys())[1]:
            codes[symbol] +=  '1'

    return pd.DataFrame({'states': states, 'probab': probab, 'code':list(codes.values()), })

def L(distr_coded): # contamos con la normalización anterior de los pesos para da lugar a frecuencias relativas
    sum = 0
    #print(np.sum(np.array(list(distr_coded['probab']))))
    for i,row in distr_coded.iterrows():
        sum += row['probab']*len(row['code'])
    return sum


def H(distr_coded):
    sum = 0
    for i,row in distr_coded.iterrows():
        sum+=row['probab']*math.log(row['probab'],2)
    return -sum

def check_shannon(distr_coded):
    h=H(distr_coded)
    l=L(distr_coded)
    return h <= l <= h+1

def code(distr_coded,word):
    word_coded=''
    codes=dict(zip(distr_coded['states'], distr_coded['code']))
    for symbol in word:
        word_coded+=codes[symbol]
    return word_coded

dc_en=calculate_codes_df(tree, distr)
print('L',L(dc_en))
print('H',H(dc_en))
print('Shannon(H<L<H+1)',check_shannon(dc_en))

distrib = distr_es 
arbol = huffman_tree(distrib)
dc_es=calculate_codes_df(arbol, distrib)
print('L',L(dc_es))
print('H',H(dc_es))
print('Shannon (H<L<H+1)',check_shannon(dc_es))

code_en=code(dc_en,'medieval')
code_es=code(dc_es,'medieval')
print('medieval in inglés:',code_en)
print('medieval en Spagnolo:',code_es)

dc_es_ord=dc_es.sort_values(by='code', ascending=True)
'''
def check_better_usual(distr_coded):
    sum = 0
    for i,row in distr_coded.iterrows():
        sum+=len(row['code'])
    N=len(distr_coded.index)
    print('suma c_i',sum)
    return sum <= N*math.log(N,2)
def check_better_usual(distr_coded,code,word):
    N=len(distr_coded.index)
    return len(code) <= len(word)*math.log(N,2)
'''
def check_better_usual(distr_coded,code):
    N=len(distr_coded.index)
    return len(code) <= N*math.log(N,2)

print('mejor usual eng',check_better_usual(dc_en,code_en))
print('mejor usual esp',check_better_usual(dc_es,code_es))

def decode(distr_coded,word):
    word_decoded=''
    codes=dict(zip(distr_coded['code'], distr_coded['states']))
    begin=0
    for end in range(len(word)+1):
        symbol=codes.get(word[begin:end])
        if symbol!=None:
            word_decoded+=symbol
            begin=end
    return word_decoded
#Operacion inversa a la codificacion de medieval
#print('decode to inglés',decode(dc_en,'11110101111110110111111000111011110100110101101110'))
#print('decodificar a Spagnolo',decode(dc_es,'11000101000010110010100111010100110101'))

print('decode',10111101101110110111011111,'to inglés:',decode(dc_en,'10111101101110110111011111'))
