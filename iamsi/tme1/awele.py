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
            estChezEnnemi = (indice_courant <= n)
        else:
            estChezEnnemi = (indice_courant > n)
    # mise a jour du camp au trait
    if trait == 'SUD':
        nouvelle_pos['trait'] = 'NORD'
    else:
        nouvelle_pos['trait'] = 'SUD'
    return nouvelle_pos

def coupJouable(position,nombre):
    jouable=False
    pleine=False
    if(nombre>=1 and nombre<=position['taille']):    
        jouable=True
    if( position['trait']== "SUD"):
        if(position['tablier'][nombre-1]>0):
            pleine=True
    if( position['trait']== "NORD"):
        if(position['tablier'][2*position['taille']-nombre]>0):
            pleine=True
    return jouable and pleine
    
def coupAutorise(position,coup):
    if(coupJouable(position,coup)):        
        pos = joueCoup(position,coup)
        if(position['trait']=='SUD'):
            if sum(pos['tablier'][0:pos['taille']-1]):
                return pos
        if(position['trait']=='NORD'):
            if sum(pos['tablier'][pos['taille']:-1+2*pos['taille']]):
                return pos
    return False
              #TO CHECKKKKKKKKKKKK §§§§§!! ! ! ! ! ! ! ! ! !       
def positionTerminale(position):
    if(position['graines']['NORD']>=25 or position['graines']['SUD']>=25):
        return True    
    if(position['trait']=='SUD'):
        for a in range(0,position['taille']):
            if(coupAutorise(position,a)):
                return False
    if(position['trait']=='NORD'):
        for a in range(position['taille'],2*position['taille']):
            if(coupAutorise(position,a)):
                return False
    return True


def moteurHumains(taille=3):
    pos=initialise(taille)    
    while(not(positionTerminale(pos))):
        affichePosition(pos)
        coup=input('Rentre ton coup, c\'est ton tour!\n')
        if(coupAutorise(pos,coup)):
            pos=joueCoup(pos,coup)
        else:
            print('essaye encore...\n')

    print("FINI")
moteurHumains(6)

'''print("------\nPartie sur un tablier reduit pour tester:")
maPosition = initialise(3)
affichePosition(maPosition)
maPosition2 = joueCoup(maPosition,1) # SUD joue
affichePosition(maPosition2)
maPosition2 = joueCoup(maPosition2,2) # NORD joue
affichePosition(maPosition2)
maPosition2 = joueCoup(maPosition2,3) # SUD joue
affichePosition(maPosition2)
maPosition2 = joueCoup(maPosition2,1) # NORD joue
affichePosition(maPosition2)
maPosition2 = joueCoup(maPosition2,2) # SUD joue
affichePosition(maPosition2)
maPosition2 = joueCoup(maPosition2,3) # NORD joue
affichePosition(maPosition2)
# ------------------------- FIN TEST'''
