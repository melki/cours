#hide.
#show col/2.

noeud(1..7).

couleurs(jaune;bleu;vert;vertCacaDoie).


voisin(1,2).
voisin(1,5).
voisin(1,6).

voisin(2,6).
voisin(2,7).
voisin(2,3).

voisin(3,4).
voisin(3,6).
voisin(3,5).
voisin(3,7).

voisin(4,5).

voisin(5,6).
voisin(5,7).

voisin(6,7).



1{col(A,B):couleurs(B)}1:-noeud(A).

:-voisin(A,B),col(A,C1),col(B,C1).




