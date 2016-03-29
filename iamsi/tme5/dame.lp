#hide.
#show queen/2.

pos(1..8).
1{queen(I,J):pos(J)}1:-pos(I).

:-queen(I1,J),queen(I2,J),pos(I1),pos(I2),pos(J),I1!=I2.
:-queen(I1,J1),queen(I2,J2),pos(I1;I2;J1;J2),I1!=I2,J1!=J2,#abs(I1-I2)==#abs(J1-J2).