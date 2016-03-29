
# coding: utf-8

# #  Awele TME 1
# ##DE BEZENAC DE TOLDI
# 

# In[9]:

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
import numpy as np

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


    


# ## EXERCICE 1

# In[16]:

import random
def coupJouable(position,nombre):
    """ POSITION * NOMBRE(COUP) -> BOOLEAN
        

        Cette fonction retourne un boolean indiquant si un coup est jouable à partir de la position donnée.
    """
    jouable=False #deviens vrai si le coup est un nombre du tablier
    pleine=False #deviens vrai si la case de départ possède plus d'une graine
    
    if(nombre>=1 and nombre<=position['taille']):    
        jouable=True
    
    #deux tests différents pour nord et sud
    if( position['trait']== "SUD"):
        if(position['tablier'][nombre-1]>0):
            #print str(position['tablier'][nombre-1]) + " graines"
            pleine=True
    if( position['trait']== "NORD"):
        if(position['tablier'][2*position['taille']-nombre]>0):
            pleine=True
    return jouable and pleine
    
def coupAutorise(position,coup):
    """ POSITION * COUP -> BOOLEAN || POSITION
    
        Cette fonction retourne un boolean si le coup n'est pas autorisé ou la position si le coup est autorisé
    """
    if(coupJouable(position,coup)):
        pos = joueCoup(position,coup)
        if(position['trait']=='NORD'):
            if sum(pos['tablier'][0:pos['taille']]): #test si le tablier est vide (au moins une graine dedans)
                return pos
        if(position['trait']=='SUD'):
            if sum(pos['tablier'][pos['taille']:2*pos['taille']]):
                return pos
    return False
              
def positionTerminale(position):
    """ POSITION  -> BOOLEAN
        

        Cette fonction retourne un boolean qui prend la valeur vrai si la position est terminale (aucun coup legal jouable) et la valeur fausse sinon
    """
    if(position['graines']['NORD']>=25 or position['graines']['SUD']>=25): 
        return True    
    for a in range(1,position['taille']+1):
        if(coupAutorise(position,a)):
                return False #on renvoie false dès qu'on obtient au moins un coup jouable
    return True


def moteurHumains(taille=3):
    """ TAILLE  -> rien
        

        Cette fonction permet à deux joueurs humains de s'affronter, les faisant jouer chacun à leur tour
    """
    pos=initialise(taille)    
    while(not(positionTerminale(pos))): #on boucle tant que le joueur peux jouer
        affichePosition(pos)
        coup = pos['taille']+1
        while(coup>pos['taille']):
            coup=input('Rentre ton coup, c\'est ton tour!\n')
        if(coupAutorise(pos,coup)):
            pos=joueCoup(pos,coup)
        else:
            print('essaye encore...\n') #cas où le joueur ne rentre pas un coup légal
    print("*** FINI ***")
    affichePosition(pos)
    if(pos['graines']['NORD'] == pos['graines']['SUD']):
        print "Match Nul"
    else:
        gagnant = "Nord" if (pos['graines']['NORD'] > pos['graines']['SUD']) else "Sud"      
        print gagnant + " remporte la manche"
        
        
def choixAleatoire(position):
    """ POSITION  -> COUP
        

        Cette fonction retourne un coup jouable aléatoire pour le cpu ou 0 si aucun coup n'est jouable à partir de la position envoyé
    """
    if(positionTerminale(position)):
        return 0
    coupJouable = []
    for a in range(1,position['taille']+1):
        if coupAutorise(position,a):
            coupJouable.append(a)
    random.shuffle(coupJouable) #on mélange le tableau des coups aléatoires
    return(coupJouable[0])

def moteurAleatoire(campCPU = "NORD"):
    """ CAMPCPU  -> rien
        

        Cette fonction permet à un joueur d'affronter un ordinateur choississant ses coup aléatoirement dans la liste des coups possible à chaque position
    """
    taille = input("Quelle taille pour cette partie ?")
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
            coup = 0 #on initialise avec un coup non légal pour pouvoir boucler
            while(coup<1 or coup>pos['taille']):
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
            
            
            
        


# ##  EXERCICE 2 miniMax

# In[21]:

def nbCase(c,p):
    """ CAMP, POSITION  -> INTEGER
        

        Cette fonction renvoie le nombre de cases avec une ou deux graines dans le camp donné avec un postion donné 
    """
        
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
    """ POSITION  -> EVALUATION
        

        Cette fonction renvoie l'évaluation d'une postion en utilisant la fonction donnée
    """
    if(positionTerminale(position)):
        if(position['graines']['NORD'] > position['graines']['SUD']):
            return -1000
        else:
            return 1000
    else:
        return 2*position['graines']['NORD']+nbCase('NORD',position) - (2*position['graines']['SUD']+nbCase('SUD',position))


    
def evalueMiniMax(position,prof,coup=1):
    """ POSITION, PROFONDEUR,COUP  -> {MEILLEUR COUP,VALEUR DE CE COUP}
        

        Cette fonction utilise l'algorithme minimax pour calculer le meilleur coup à partir d'une position et d'une profondeur données
    """
    if prof == 0 or positionTerminale(position):
        
        return {'coup':coup,'valeur':evaluation(position)}
    if position['trait'] == 'NORD':
        bestValue = - float('inf')
        bestCoup = 0
        for a in range(1,position['taille']+1):
            if(coupAutorise(position,a)):
                p = clonePosition(position)
                p = joueCoup(position,a)
                e = evalueMiniMax(p,prof-1,a)
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
                e = evalueMiniMax(p,prof-1,a)
                
                if bestValue >= e['valeur']:
                    bestValue = e['valeur']
                    bestCoup = a
        return {'coup':bestCoup,'valeur':bestValue}

                
def choixMinimax(position,prof):
    """ POSITION,PROFONDEUR  -> COUP
        

        Cette fonction renvoie le coup choisi par la fonction evalueMiniMax ou 0 si la position donnée est terminale
    """
    if(positionTerminale(position)):
        return 0
    else:
        coup = evalueMiniMax(position,prof)
        return coup['coup']
    
def moteurMiniMax(campCPU="NORD",prof=3):
    """ CAMPCPU,PROFONDEUR   -> Rien
        

        Cette fonction permet à l'utilisateur d'affronter un cpu utilisant l'algorithme du minimax pour choisir ses coups.
    """
    taille = input("Quelle taille pour cette partie ?")
    pos = initialise(taille)
    affichePosition(pos)
    print '*** on commence ***'
    while(not(positionTerminale(pos))):
        
        if(campCPU == pos['trait']):
            coup = choixMinimax(pos,prof)
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
            
      


# ## EXERCICE 3: ALPHA-BETA

# In[22]:


def evalueAlphaBeta(position,prof,i,alpha,beta):
    """ POSITION, PROFONDEUR,COUP,ALPHA,BETA  -> {MEILLEUR COUP,VALEUR DE CE COUP}
        

        Cette fonction utilise l'algorithme alpha beta pour calculer le meilleur coup à partir d'une position et d'une profondeur données 
    """
    if prof==0 or positionTerminale(position): #si la position est terminale, ou le parcours de l'arbre est fini
        return {'coup':1,'valeur':evaluation(position)}
    else:
        bestCoup=0
        j=position['taille']
        if position['trait']=='NORD':
            #position est MAX
            i=1
            while(i<=j and alpha<beta): #tant qu'on ne peut pas élaguer
                if(coupAutorise(position,i)):
                    p=clonePosition(position)
                    p=joueCoup(position,i)
                    e=evalueAlphaBeta(p,prof-1,i,alpha,beta)
                    if(alpha<e['valeur']): #on stocke l'alpha min
                        bestCoup=i
                        alpha=e['valeur']                    
                i+=1
            return {'coup':bestCoup,'valeur':alpha}
        else:
            #position est MIN
            i=1
            while(i<=j and alpha<beta): #tant qu'on ne peut pas élaguer
                if(coupAutorise(position,i)):
                    p=clonePosition(position)
                    p=joueCoup(position,i)
                    e=evalueAlphaBeta(p,prof-1,i,alpha,beta)
                    if(beta>e['valeur']): #on stocke le beta max
                        bestCoup=i
                        beta=e['valeur']
                i+=1
            return {'coup':bestCoup,'valeur':beta}

def choixAlphaBeta(position,prof):
    """ POSITION,PROFONDEUR  -> COUP
        

        Cette fonction renvoie le coup choisi par la fonction evalueAlphaBeta ou 0 si la position donnée est terminale
    """
    if(positionTerminale(position)):
        return 0
    else:
        coup=evalueAlphaBeta(position,prof,1,-np.inf,np.inf)
        return coup['coup']

def moteurAlphaBeta(campCPU="NORD",prof=3):
    """ CAMPCPU,PROFONDEUR   -> Rien
        

        Cette fonction permet à l'utilisateur d'affronter un cpu utilisant l'algorithme de l'apha beta pour choisir ses coups.
    """
    taille = input("Quelle taille pour cette partie ?")
    pos = initialise(taille)
    affichePosition(pos)
    print '*** on commence ***'
    while(not(positionTerminale(pos))):
        
        if(campCPU == pos['trait']):
            coup = choixAlphaBeta(pos,prof)
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




# #Approfondissement
# 
# Afin de trouver les paramètres optimaux de la fonction d'évaluation nous avons mis en place un algorithme génétique, ici avec un minimax (très long) mais facilement adaptable avec un algorithme alpha beta.
# 
# On a décomposé la fonction d'évaluation en 6 paramètres A,B,C,D,E et F (cf evaluationCpu).
# 
# Ensuite on procède aux étapes habituelles d'un algorithme génétique :
#    
#    * Création de la population de la génération 0 `genererPopulation(nombre d'indidividus)`
#    * Création des individus de cettes population `genererIndividu`
#    * On fait une boucle sur le nombre de génération voulu (ex 50)
#        * On fait s'affrontter les individus par "poules" pour alléger les calculs, ex: sur un population de 50 on fait 5 poules de 10. Dans chaque poules tout les individus s'affrontent en utilisant leurs propre fonction d'évaluation : `tournoi`
#        * On récupère les parents : les x premiers de chaque poules.
#        * Les parents se "reproduisent", les autres sont éliminés, il y une petite probabilité qu"un des parents mue: `reproduction` et `mutation`
#    * C'est la fonction `nouvelleGeneration` qui gère tout cela
#    * Une fois qu'on a la génération finale, on les fait s'affronter une dernière fois et on récupère le "Champion"
#    
# On a remarqué un pourcentage de réussite allant jusqu'à 70% (affrontement du champion contre 1000 individus utilisant minimax avec une fonction d'évalutation alléatoire) avec certains paramètre, mais cela reste variable et la méthode n'a pas été très concluante.
# 
# Les fonction xxxxCpu sont les fonctions usuelles adaptées à une fonction d'évalutation variable.
# 
# Le meilleur champion trouvé est :
#     {'A': 0.5059132062875165, 'C': 0.6081366767125125, 'B': 0.596495474468575, 'E': 0.46028412950161013, 'D': 0.7411628325957265, 'F': 0.3750175831732554, 'generation': 0, 'nbVictoire': 46, 'camp': 'SUD'}
#     

# In[28]:

def autreCamp(a):
    if a == "NORD":
        return "SUD"
    else:
        return "NORD"
def evaluationCpu(position,cpu):
    
    if(positionTerminale(position)):
        if(position['graines']["SUD"] > position['graines']["NORD"]):
            return cpu["A"]*(-10000)
        else:
            return cpu["B"]*(10000)
    else:
        return cpu["C"]*2*position['graines']["NORD"]+cpu["D"]*nbCase("NORD",position) - (cpu["E"]*2*position['graines']["SUD"]+cpu["F"]*nbCase("SUD",position))


    
def evalueMiniMaxCpu(position,prof,campCPU,cpu,coup=1):
   
    if prof == 0 or positionTerminale(position):
        
        return {'coup':coup,'valeur':evaluationCpu(position,cpu)}
    if position['trait'] == "NORD":
        bestValue = - float('inf')
        bestCoup = 0
        for a in range(1,position['taille']+1):
            if(coupAutorise(position,a)):
                p = clonePosition(position)
                p = joueCoup(position,a)
                e = evalueMiniMaxCpu(p,prof-1,campCPU,cpu,a)
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
                e = evalueMiniMaxCpu(p,prof-1,campCPU,cpu,a)
                
                if bestValue >= e['valeur']:
                    bestValue = e['valeur']
                    bestCoup = a
        return {'coup':bestCoup,'valeur':bestValue}

                
def choixMinimaxCpu(position,prof,campCPU,cpu):
    if(positionTerminale(position)):
        return 0
    else:
        coup = evalueMiniMaxCpu(position,prof,campCPU,cpu)
        return coup['coup']

def moteurCpuVsCpu(cpu1,cpu2,prof = 3,taille = 6):
    coupSansPrise = 0
    al = random.random()
    if(al > .5):
        cpu1['camp'] = 'SUD'
        cpu2['camp'] = 'NORD'
    else:
        cpu2['camp'] = 'SUD'
        cpu1['camp'] = 'NORD'
    i=0
    pos = initialise(taille)
    while(not(positionTerminale(pos))):
        graineNord = pos['graines']['NORD']
        graineSud = pos['graines']['SUD']
        i+=1
        if i%500 == 0:
            print i
        if(cpu1['camp'] == pos['trait']):
            coup = choixMinimaxCpu(pos,prof,cpu1['camp'],cpu1)
            pos = joueCoup(pos,coup)
        else:
            coup = choixMinimaxCpu(pos,prof,cpu2['camp'],cpu2)
            pos = joueCoup(pos,coup)
        if(graineNord == pos['graines']['NORD'] and graineSud == pos['graines']['SUD']):
            coupSansPrise +=1
        else:
            coupSansPrise = 0
            
        if coupSansPrise>10:
            break
    gagnant = "NORD" if (pos['graines']['NORD'] > pos['graines']['SUD']) else "SUD"      
    if(cpu2['camp']==gagnant):
        return 2
    else:
        return 1
        
import numpy as np
from operator import itemgetter

def genererIndividu():
    param = [random.random() for a in range(6) ]
    
    return {'camp':"","A":param[0],"B":param[1],"C":param[2],"D":param[3],"E":param[4],"F":param[5],'generation':0,'nbVictoire':0}

def genererPopulation(n):
    pop = []
    for i in range(n):
        pop.append(genererIndividu())
    print 'population crée !'
    return pop

def tournoi(pop,nbSelec):
    for i in range(len(pop)):
        
        for j in range(i,len(pop)):
            if(i!=j):
                if(moteurCpuVsCpu(pop[i],pop[j])==1):
                    pop[i]['nbVictoire']+=1
                else:
                    pop[j]['nbVictoire']+=1
    newPop = sorted(pop, key=itemgetter('nbVictoire'),reverse=True) 
    return newPop[0:nbSelec]

def reproduction(parent1,parent2):
    enfant = {'camp':"","A":0,"B":0,"C":0,"D":0,"E":0,"F":0,'generation':0,'nbVictoire':0}
    for l in ['A','B','C','D','E','F']:
        alpha = .5
        enfant[l] = parent1[l]*alpha + parent2[l]*(1-alpha)
    
    return enfant

def mutation(individu):
    parametreAChanger = random.randrange(0,6)
    letters = ['A','B','C','D','E','F']
    mutant = individu
    mutant[letters[parametreAChanger]] = random.random()
    return mutant

def nouvelleGeneration(parents,n):
    nouvelleGen = []
    for p in parents:
        p['nbVictoire'] = 0
        nouvelleGen.append(p)
        a = random.random()
        if a > .95:
            nouvelleGen.append(mutation(p))
    while len(nouvelleGen) < n:
        p1 = random.randrange(0,len(parents))
        p2 = random.randrange(0,len(parents))
        if p1 != p2:
            bebe = reproduction(parents[p1],parents[p2])
            
            nouvelleGen.append(bebe)
    return nouvelleGen
    


# ### example de l'algorithme 

# In[ ]:

pop = genererPopulation(10) #génération de la population 0 (ici valeur très faible pour exemple, dans l'idéal mettre des valeurs bien plus grandes et utiliser alpha beta)

for i in range(10): #une itération par génération
    
    print "generation : " + str(i)
    pop1 = tournoi(pop[0:5],1)
    pop2 = tournoi(pop[5:10],1) #séparation en poule
    pop = nouvelleGeneration([pop1[0],pop2[0]],10) #création de la nouvelle génération

winner = tournoi(pop,1) #dernier tournoi pour trouver le champion
winner = winner[0]
print winner   

pop = genererPopulation(1000) #batterie de test
score = 0
for i in range(len(pop)):
    if i%100 == 0:
        print str(i)+" matchs !"
        
        print str(score/((i+1.)/100))+"% de réussite, pas mal la bète !"
    res = moteurCpuVsCpu(winner,pop[i])
    if res == 1:
        score+=1
print str(score/10)+"% de réussite, pas mal la bète !"

