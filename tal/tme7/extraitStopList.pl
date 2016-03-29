#!/usr/bin/perl

# lance l'étiquetage du fichier passe en parametre f, cree un fichier de meme nom complete par .tag
# ecrit les formes de la caracteristique dans le fichier de sortie "f.mot/f.lemme"
$script_name = $0;

sub usage
{
    die "usage: perl $script_name <input-file> {mot|lemme} \n";
}

if(@ARGV != 2)
{
    usage;
}

$file  = $ARGV[0];
$etude = $ARGV[1];

$POSgarde = '\)|\$|\#|CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|LS|MD|PDT|POS|PP|RB|RBR|RBS|RP|SENT|SYM|TO|UH|VB|VBD|VBG|VBN|VBP|VBZ|WDT|WP|WP$|WRB';
#'\)|\$|\#|CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|LS|MD|PDT|POS|PP|RB|RBR|RBS|RP|SENT|SYM|TO|UH|VB|VBD|VBG|VBN|VBP|VBZ|WDT|WP|WP$|WRB|NP|NPS'; ## que les noms communs
#  tous les noms : '\)|\$|\#|CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|LS|MD|PDT|POS|PP|RB|RBR|RBS|RP|SENT|SYM|TO|UH|VB|VBD|VBG|VBN|VBP|VBZ|WDT|WP|WP$|WRB'
# nom, verbe, adjectif '\)|\$|\#|CC|CD|DT|EX|FW|IN|LS|MD|PDT|POS|PP|RB|RBR|RBS|RP|SENT|SYM|TO|UH|WDT|WP|WP$|WRB'

if($etude !~ /^mot$/i &&  $etude !~ /^lemme$/i)
{
    usage;
}

if ($POSgarde eq "") 
{
	print "toutes les categories gardees\n";
}

if ($file =~ /tag$/) { die "un fichier tagge $file\n"};

$reponse = '';
if(-e "$file.tag")
{
    do
    {
        print "Le fichier $file.tag existe déjà, voulez-vous relancer tree-tagger ? ";
        chomp($reponse = <STDIN>);
    }while($reponse !~ /^y(es)?$/i && $reponse !~ /^no?$/i );
}

if ($reponse =~ /^y(es)?$/i || $reponse eq '')
{
    system( "tree-tagger-english $file > $file.tag" )
};


open (FI, "<$file.tag") || die "pb ouverture fichier $file.tag\n";
#print "fichier lu: $file.tag\n";
# extension selon la caracteristique etudiee : .mot, .lemme
open (FOUT, ">$file.$etude");
while (<FI>) {
    if ( /(.*)\t(.*)\t(.*)/) {
    	$mot = $1; 
        $cat = $2; 
        $lemme = $3;
    	if ($lemme eq "<unknown>") {
    	    $lemme = $mot;
    	}
	if ($lemme eq "@card") {
    	    $lemme = $mot;
    	}
    	$cat =~ s/^([^:]*):.*$/$1/;
        
        # variable selon la caracteristique etudiee: $mot, $lemme, $cat
    	print FOUT "$mot\n"    if (($etude =~ /^mot$/i) && ($cat =~ /$POSgarde/));
#    	print FOUT "$cat\n"    if($etude =~ /^cat$/i);
    	print FOUT "$lemme\n"  if(($etude =~ /^lemme$/i) && ($cat =~ /$POSgarde/));
    }
}
close FI;
close FOUT;
