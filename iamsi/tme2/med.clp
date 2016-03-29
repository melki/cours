(defrule my_init
	 (initial-fact)
=>
	(watch facts)
	(watch rules)
	
	(assert (taches_rouges))
	(assert (bouton peu))
	(assert (sensation_de_froid oui))
	(assert (fievre forte))
	(assert (yeux_douleureux))
	(assert (amygdales_rouges))
	(assert (peau pele seche))

)


(defrule eruption_cutane
	 (bouton ?x)
=>
	(assert (eruption_cutane))
)


(defrule exantheme1
	 (rougeur)
=>
	(assert (exantheme))
)

(defrule exantheme2
	 (eruption_cutane )
=>
	(assert (exantheme))
)

(defrule etat_febrile1
	 (fievre forte)
=>
	(assert (etat_febrile))
)

(defrule etat_febrile2
	 (sensation_de_froid oui)
=>
	(assert (etat_febrile))
)

(defrule signe_suspect
	 (amygdales_rouges)
	 (taches_rouges)
	 (peau pele ?x)
=>
	(assert (etat_febrile))
)

(defrule signe_suspect
	 (amygdales_rouges)
	 (taches_rouges)
	 (peau pele ?x)
=>
	(assert (etat_febrile))
)

(defrule rougeole1
	 (etat_febrile)
	 (yeux_douleureux)
	 (exantheme)
=>
	(assert (rougeole oui))
)

(defrule rougeole2
	 (signe_suspect)
	 (fievre forte)
=>
	(assert (rougeole oui))
)

(defrule rougeole2
	 (bouton peu)
	 (fievre peu)
=>
	(assert (rougeole non))
)


(defrule douleur1
	 (yeux_douleureux)
=>
	(assert (douleur))
)

(defrule douleur2
	 (dos_douleureux)
=>
	(assert (douleur))
)

(defrule grippe
	 (dos_douleureux)
	 (etat_febrile)
=>
	(assert (grippe))
)

(defrule varicelle
	 (rougeole non)
	 (demangeaisons fortes)
	 (pustules oui)
=>
	(assert (varicelle))
)

(defrule rubeole
	 (rougeole non)
	 (peau ?x seche)
	 (ganglions)
	 (pustules non)
	 (sensation_de_froid non)
=>
	(assert (rubeole))
)

