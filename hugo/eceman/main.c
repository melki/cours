#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

int X1 = 7;
int Y1 = 1;

int PERDU = FALSE;
char TAB[15][19];
char CASEDAVANT;

void gotoligcol( int lig, int col )
{
 // ressources
 COORD mycoord;

 mycoord.X = col;
 mycoord.Y = lig;
 SetConsoleCursorPosition( GetStdHandle( STD_OUTPUT_HANDLE ), mycoord );
}
int menu()
{
    int choix=0;
    printf ("saisissez votre niveau");

        printf("Menu :\n\n");
        printf("1: le niveau 1: il faut atteindre la porte en brisant la glace fine et en ne passant pas par l eau\n\n");
        printf("2: le niveau 2: il faut atteindre la porte en brisant la glace epaisse en passant deux fois dessus et en brisant la glace legere en passant une fois dessus. On ne peut toujours pas passer par l eau\n\n");
        printf("3: le niveau 3: Vous utiliserez une tondeuse que ECEMAN peut pousser et qui brise toute la glace sur sa lancee jusqu a s arreter sur un mur. Vous utiliserez aussi des briques de mur qu ECEMAN peut pousser pour les deplacer dans les 4 directions. Vous disposerez aussi de« potions de legerete » qui permettent à ECEMAN de passer sur la glace sans la briser\n\n");
        printf("4: le niveau 4: Le niveau 4 ajoute un tunnel dans lequel ECEMAN doit passer pour passer d une zone fermee du tableau a une autre zone fermee qui contient entre autres la porte de sortie (le tunnel est virtuel, la porte d entree du tunnel se trouve dans une des zones fermees et lorsque ECEMAN l emprunte il ressort par l autre porte du tunnel dans une autre zone fermee, sans possibilite de faire marche arriere).\n\n");
        printf("5: le niveau 5: fera apparaitre, en plus, un ou plusieurs ennemis qui « balayent le terrain » en se deplaçant de gauche a droite ou de haut en bas en changeant de direction au contact d un mur ou d un obstacle. Si ECEMAN touche un ennemi, le tableau est perdu et doit etre recommence.\n\n");
        printf("quel est votre choix ?");
        scanf("%d",&choix);

    return choix;


}
void affiche()
{
int i = 0;
int j= 0;

for(i=0; i<15; i++)
              {
              for(j=0; j<19 ;j++)
                 {
                  printf("%c",TAB[i][j]);
                 }
             printf("\n");

             }
printf("\n");
             printf("\n");
             printf("\n");
             printf("\n");
             printf("\n");
             printf("\n");
             printf("\n");
}


void niveau1()
// décalaration de variables

{

int j;
int i;
// traitement place tableau

for(i=0; i<15; i++)
{
    for(j=0; j<19; j++)
       {
        TAB[i][j]='#';
       }
       printf("\n");
       }
//traitement glace
for(i=0; i<15; i++)
           {
            TAB[7][j-2]='O';
           for(j=0; j<19; j++)
                {
                TAB[7][j-2]='O';
                }
           }
       //traitement mur


           TAB[7][0]='M';
           TAB[7][18]='M';

       for(j=0; j<19; j++)
          {
           TAB[6][j]='M';
           TAB[8][j]='M';
          }
        //traitement personnage


        //traitement sortie
        TAB[7][17]='E';
      CASEDAVANT = TAB[7][1];
      TAB[7][1]='B';



}

void niveau2()
// décalaration de variables

{



int j;
int i;
// traitement place tableau

for(i=0; i<15; i++)
{
    for(j=0; j<19; j++)
       {
        TAB[i][j]='#';
       }
       printf("\n");
       }
//traitement glace

for(i=0; i<15; i++)
           {

           for(j=0; j<19; j++)
                {
                TAB[8][j-1]='O';
                TAB[7][j-2]='O';
                TAB[6][j-2]='O';
                }
           }
       //traitement mur

           TAB[7][1] = 'X';

           TAB[7][0]='M';
           TAB[6][0]='M';
           TAB[8][0]='M';
           TAB[6][18]='M';
           TAB[7][18]='M';
           TAB[8][18]='M';

       for(j=0; j<19; j++)
          {
           TAB[5][j]='M';
           TAB[9][j]='M';
          }
        //traitement personnage


        //traitement sortie
        TAB[7][17]='E';
      CASEDAVANT = TAB[7][1];
      TAB[7][1]='B';



}







void jeumatrice()
{
    // Déclaration des variables

    int i=0;
    int j=0;
    int x1,y1,x2,y2;
    int vx = 0;
    int vy = 0;
    char key=0;
    // Définition de la position de départ

    //matrice [nb1][nb2]='B';

    // Déplacement du personnage

      affiche();
    while (key != 'n' && PERDU == FALSE)
{
        if (kbhit())
        {

        key=getch();
        switch (key)
        {

            case 'z':
                    vx = -1;
                    vy=0;
            break;
            case 'd':
                     vy = 1;
                     vx = 0;
            break;
            case 'q':
                     vy = -1;

                     vx = 0;
            break;
            case 's':
                     vx = 1;
                     vy=0;
            break;
        }
           x1 = X1;
           y1 = Y1;
           x2=X1+ vx;
           y2=Y1 + vy;

        if(TAB[x2][y2] == ' ')
        {
            PERDU = TRUE;
        }

        else if  (TAB[x2][y2] == 'M')
            {
                TAB[x1][y1]='B';
            }
        else
          {
                if(CASEDAVANT =='O')
                {
                    TAB[x1][y1] = ' ';
                }
                if(CASEDAVANT =='X')
                {
                    TAB[x1][y1] = 'O';
                }
                 if(CASEDAVANT =='E')
                {
                    TAB[x1][y1] = 'E';
                }


                CASEDAVANT = TAB[x2][y2];
                TAB[x2][y2]='B';

                X1 = x2;
                Y1 = y2;




          }
        if(CASEDAVANT == 'E')
        {

            int i,j;
            int obsRestant=0;
         for(i=0; i<15; i++)
              {
              for(j=0; j<19 ;j++)
                 {
                   if(TAB[i][j] == 'O' || TAB[i][j] == 'X')
                   {

                       obsRestant ++;
                   }
                 }
             printf("\n");

             }
             if( obsRestant == 0)
            {
                printf("Vous avez gagne le niveau, Bravo !\n");
                return 0;
             }





        }


            affiche();

        }


}
    if(PERDU == TRUE)
    {
        printf("Vous avez perdu au revoir !");
    }
}




int main()
{


   switch (menu())
   {
   case 1:
        system ("cls");
        printf("vous avez choisi le niveau 1\n");

        niveau1();
        jeumatrice();


        break;
    case 2:

        printf("vous avez choisi le niveau 2");

        niveau2();
        jeumatrice();


        break;
    case 3:
        system ("cls");
        printf("vous avez choisi le niveau 3");
        break;
    case 4:
        system ("cls");
        printf("vous avez choisi le niveau 4");
        break;
    case 5:
        system ("cls");
        printf("vous avez choisi le niveau 5");
    break;

   }


   return 0;



}




