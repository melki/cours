
# coding: utf-8

# In[179]:

# - - - - - - - - - - -
# IAMSI - 2016
# joueur d'Awele
# - - - - -
# REM: ce programme a ete ecrit en Python 3.4
# 
# En salle machine : utiliser la commande "python3"
# - - - - - - - - - - -

# - - - - - - - - - - - - - - - INFORMATIONS BINOME
# GROUPE DE TD : 1
# NOM, PRENOM  : DE BEZENAC EMMANUEL
# NOM, PRENOM  : DE TOLDI MELCHIOR
# - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - TYPES UTILISES
# POSITION : dictionnaire non pleine qui contient differentes informations sur
#            une position d'Awele, associees au nom de leur champ.
# COUP : valeur entiere comprise entre 1 et le nombre de colonnes du tablier

# - - - - - - - - - - - - - - - INITIALISATION
def initialise(n):
    """ int -> POSITION
        Hypothese : n > 0
        initialise la position de depart de l'awele avec n colonnes avec 4 dans chaque case.
    """
    position = dict()                                 # initialisation
    position['tablier'] = [4 for k in range(0, 2*n)]  # on met 4 graines dans chaque case
    position['taille'] = n                            # le nombre de colonnes du tablier
    position['trait'] = 'SUD'                         # le joueur qui doit jouer: 'SUD' ou 'NORD'
    position['graines'] = {'SUD':0, 'NORD':0}         # graines prises par chaque joueur
    return position

# - - - - - - - - - - - - - - - AFFICHAGE (TEXTE)
def affichePosition(position):
    """ POSITION ->
        affiche la position de facon textuelle
    """
    print('* * * * * * * * * * * * * * * * * * * *')
    n = position['taille']
    buffer = 'col:'
    for i in range(0,n):
        buffer += ' ' + str(i+1) + ' \t'
    print(buffer)
    print('\t\tNORD (prises: '+str(position['graines']['NORD'])+')')
    print('< - - - - - - - - - - - - - - -')
    buffer = ''
    for i in range(2*n-1,n-1,-1):   # indices n..(2n-1) pour les cases NORD
        buffer += '\t[' + str(position['tablier'][i]) + ']'
    print(buffer)
    buffer = ''
    for i in range(0,n):            # indices 0..(n-1) pour les cases SUD
        buffer += '\t[' + str(position['tablier'][i]) + ']'
    print(buffer)
    print('- - - - - - - - - - - - - - - >')
    print('\t\tSUD (prises: '+str(position['graines']['SUD'])+')')
    print('-> camp au trait: '+position['trait']);

# - - - - - - - - - - - - - - - CLONAGE
import copy
def clonePosition(position):
    """ POSITION -> POSITION
        retourne un clone de la position
        (qui peut etre alors modifie sans alterer l'original donc).
    """
    leclone = dict()
    leclone['tablier'] = copy.deepcopy(position['tablier'])
    leclone['taille']  = position['taille']
    leclone['trait']   = position['trait']
    leclone['graines'] =  copy.deepcopy(position['graines'])
    return leclone

# - - - - - - - - - - - - - - - JOUE UN COUP
def joueCoup(position,coup):
    """ POSITION * COUP -> POSITION
        Hypothese: coup est jouable.

        Cette fonction retourne la position obtenue une fois le coup joue.
    """
    nouvelle_pos = clonePosition(position)   # on duplique pour ne pas modifier l'original
    n = nouvelle_pos['taille']
    trait = nouvelle_pos['trait']
    # on transforme coup en indice
    if trait == 'SUD':
        indice_depart = coup-1
    else:
        indice_depart = 2*n-coup
    # retrait des graines de la case de depart
    nbGraines = nouvelle_pos['tablier'][indice_depart]
    nouvelle_pos['tablier'][indice_depart] = 0
    # on seme les graines dans les cases a partir de celle de depart
    indice_courant = indice_depart
    while nbGraines > 0:
        indice_courant = (indice_courant + 1) % (2*n)
        if (indice_courant != indice_depart):              # si ce n'est pas la case de depart
            nouvelle_pos['tablier'][indice_courant] += 1   # on seme une graine
            nbGraines -= 1
    # la case d'arrivee est dans le camp ennemi ?
    if (trait == 'NORD'):
        estChezEnnemi = (indice_courant < n)
    else:
        estChezEnnemi = (indice_courant >= n)
    # realisation des prises eventuelles
    while estChezEnnemi and (nouvelle_pos['tablier'][indice_courant] in range(2,4)):
        nouvelle_pos['graines'][trait] += nouvelle_pos['tablier'][indice_courant]
        nouvelle_pos['tablier'][indice_courant] = 0
        indice_courant = (indice_courant - 1) % (2*n)
        if (trait == 'NORD'):
            estChezEnnemi = (indice_courant < n)
        else:
            estChezEnnemi = (indice_courant >= n)
    # mise a jour du camp au trait
    if trait == 'SUD':
        nouvelle_pos['trait'] = 'NORD'
    else:
        nouvelle_pos['trait'] = 'SUD'
    return nouvelle_pos


    


# ##EXERCICE 1

# In[180]:

import random
def coupJouable(position,nombre):
    jouable=False
    pleine=False
    if(nombre>=1 and nombre<=position['taille']):    
        jouable=True
    if( position['trait']== "SUD"):
        if(position['tablier'][nombre-1]>0):
            #print str(position['tablier'][nombre-1]) + " graines"
            pleine=True
    if( position['trait']== "NORD"):
        if(position['tablier'][2*position['taille']-nombre]>0):
            #print str(position['tablier'][2*position['taille']-nombre]) + " graines"
            pleine=True
    #print 'pleine : ' + str(nombre) + "  "+str(pleine) 
    return jouable and pleine
    
def coupAutorise(position,coup):
    if(coupJouable(position,coup)):
        #print 'jouable coup : ' +str(coup)
        pos = joueCoup(position,coup)
        if(position['trait']=='NORD'):
            if sum(pos['tablier'][0:pos['taille']]):
                return pos
        if(position['trait']=='SUD'):
            if sum(pos['tablier'][pos['taille']:2*pos['taille']]):
                return pos
    #print "NON AUTORISE : coup "+str(coup)        
    return False
              
def positionTerminale(position):
    if(position['graines']['NORD']>=25 or position['graines']['SUD']>=25):
        return True    
    for a in range(1,position['taille']+1):
        if(coupAutorise(position,a)):
                return False
#         else:
#             print "le coup "+str(a)+" n'est pas autorisé pour "+str(position['trait'])
    return True


def moteurHumains(taille=3):
    pos=initialise(taille)    
    while(not(positionTerminale(pos))):
        affichePosition(pos)
        coup = pos['taille']+1
        while(coup>pos['taille']):
            coup=input('Rentre ton coup, c\'est ton tour!\n')
        if(coupAutorise(pos,coup)):
            pos=joueCoup(pos,coup)
        else:
            print('essaye encore...\n')
    print("*** FINI ***")
    affichePosition(pos)
    if(pos['graines']['NORD'] == pos['graines']['SUD']):
        print "Match Nul"
    else:
        gagnant = "Nord" if (pos['graines']['NORD'] > pos['graines']['SUD']) else "Sud"      
        print gagnant + " remporte la manche"
def choixAleatoire(position):
    if(positionTerminale(position)):
        return 0
    coupJouable = []
    for a in range(1,position['taille']+1):
        if coupAutorise(position,a):
            coupJouable.append(a)
    random.shuffle(coupJouable)
    return(coupJouable[0])

def moteurAleatoire(taille,campCPU = "NORD"):
    
    pos = initialise(taille)
    affichePosition(pos)
    print '*** on commence ***'
    while(not(positionTerminale(pos))):
        
        if(campCPU == pos['trait']):
            coup = choixAleatoire(pos)
            print 'CPU joue la case '+ str(coup)
            pos = joueCoup(pos,coup)
            affichePosition(pos)
        else:
            coup = pos['taille']+1
            while(coup>pos['taille']):
                coup=input('Rentre ton coup, c\'est ton tour!\n')
            if(coupAutorise(pos,coup)):
                pos=joueCoup(pos,coup)
            else:
                print('essaye encore...\n')  
            affichePosition(pos)
    print("*** FINI ***")
    
    if(pos['graines']['NORD'] == pos['graines']['SUD']):
        print "Match Nul"
    else:
        gagnant = "NORD" if (pos['graines']['NORD'] > pos['graines']['SUD']) else "SUD"      
        print gagnant + " remporte la manche"
        if campCPU == gagnant:
            print 'perdre contre cette IA c\' est un peu la honte...'
            
            
            
        


#    ##EXERCICE 2

# In[181]:

def nbCase(c,p):
    nb = 0
    if(c == 'NORD'):
        for i in range(p['taille'],2*p['taille']):
            if(p['tablier'][i]==1 or p['tablier'][i]==2):
                nb+=1
    if(c == 'SUD'):
        for i in range(0,p['taille']):
            if(p['tablier'][i]==1 or p['tablier'][i]==2):
                nb+=1
    
    return nb

def evaluation(position):
    
    if(positionTerminale(position)):
        if(pos['graines']['NORD'] > pos['graines']['SUD']):
            return -1000
        else:
            return 1000
    else:
        return 2*position['graines']['NORD']+nbCase('NORD',position) - (2*position['graines']['SUD']+nbCase('SUD',position))


    
def evalueMiniMax(position,prof,campCPU,coup=1):
   
    if prof == 0 or positionTerminale(position):
        
        return {'coup':coup,'valeur':evaluation(position)}
    if position['trait'] == campCPU:
        bestValue = - float('inf')
        bestCoup = 0
        for a in range(1,position['taille']+1):
            if(coupAutorise(position,a)):
                p = clonePosition(position)
                p = joueCoup(position,a)
                e = evalueMiniMax(p,prof-1,campCPU,a)
                if bestValue <= e['valeur']:
                    bestValue = e['valeur']
                    bestCoup = a
        return {'coup':bestCoup,'valeur':bestValue}

                
    else:
        bestValue = float('inf')
        bestCoup = 0
        
        for a in range(1,position['taille']+1):
            if(coupAutorise(position,a)):
                
                p = clonePosition(position)
                p = joueCoup(position,a)
                e = evalueMiniMax(p,prof-1,campCPU,a)
                
                if bestValue >= e['valeur']:
                    bestValue = e['valeur']
                    bestCoup = a
        return {'coup':bestCoup,'valeur':bestValue}

                
def choixMinimax(position,prof,campCPU):
    if(positionTerminale(position)):
        return 0
    else:
        coup = evalueMiniMax(position,prof,campCPU)
        return coup['coup']
    
def moteurMiniMax(campCPU="NORD",prof=3):
    taille = input("Quelle taille pour cette partie ?")
    pos = initialise(taille)
    affichePosition(pos)
    print '*** on commence ***'
    while(not(positionTerminale(pos))):
        
        if(campCPU == pos['trait']):
            coup = choixMinimax(pos,prof,campCPU)
            print 'CPU joue la case '+ str(coup)
            pos = joueCoup(pos,coup)
            affichePosition(pos)
        else:
            coup = pos['taille']+1
            while(coup>pos['taille']):
                coup=input('Rentre ton coup, c\'est ton tour!\n')
            if(coupAutorise(pos,coup)):
                pos=joueCoup(pos,coup)
            else:
                print('essaye encore...\n')  
            affichePosition(pos)
    print("*** FINI ***")
    
    if(pos['graines']['NORD'] == pos['graines']['SUD']):
        print "Match Nul"
    else:
        gagnant = "NORD" if (pos['graines']['NORD'] > pos['graines']['SUD']) else "SUD"      
        print gagnant + " remporte la manche"
        if campCPU == gagnant:
            print 'perdre contre cette IA c\' est un peu la honte...'
            
      


# In[182]:



# In[ ]:




# In[183]:





# In[ ]:




# In[ ]:




# In[ ]:



