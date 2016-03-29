#hide.
#show num/3.

var(1..9).
pos(0..8,0..8).

1{num(X,Y,V):var(V)}1:-pos(X,Y).
num(1,0,7).
num(0,1,8).
num(1,2,6).
num(2,2,3).

num(0,4,6).
num(0,5,9).
num(1,5,2).

num(0,7,4).
num(0,8,3).
num(1,6,5).
num(2,7,1).
num(2,8,6).




num(3,0,6).
num(5,0,3).
num(4,1,9).
num(3,2,2).
num(4,2,1).

num(4,3,3).
num(3,4,9).
num(3,5,4).
num(5,5,1).

num(3,6,8).
num(3,7,3).
num(4,7,6).
num(5,7,2).
num(3,8,1).



num(7,0,1).
num(8,0,4).
num(6,2,8).

num(7,3,5).
num(8,3,9).
num(6,4,4).
num(8,4,1).
num(8,5,8).
num(8,6,3).
num(6,7,7).
num(7,7,8).
num(6,8,9).
num(8,8,2).



:-num(X,Y,V),num(X,Z,V),Y!=Z.
:-num(X,Y,V),num(Z,Y,V),X!=Z.
:-num(X1,Y1,V),num(X2,Y2,V),X1/3==X2/3,Y1/3==Y2/3,X1!=X2.
:-num(X1,Y1,V),num(X2,Y2,V),X1/3==X2/3,Y1/3==Y2/3,Y1!=Y2.