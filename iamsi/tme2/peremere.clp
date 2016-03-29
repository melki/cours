(defrule my_init
	 (initial-fact)
=>
	(watch facts)
	(watch rules)

	(assert (pere claire jean))
	(assert (pere bob jean))
	(assert (pere yves bob))
	(assert (mere yves zoe))
	(assert (mere luc claire))
	(assert (mere alain claire))
)
(defrule parent_1
	 (pere ?x ?y)
=>
	(assert (parent ?x ?y))
)

(defrule parent_2
	 (mere ?x ?y)
=>
	(assert (parent ?x ?y))
)

(defrule grand_pere
	 (parent ?x ?y)
	 (pere ?y ?z)
=>
	(assert (grand_pere ?x ?z))
)

(defrule frere_et_soeur
	 (parent ?x ?y)
	 (parent ?u ?y)
	 (test (neq ?x ?u))
=>
	(assert (frere_et_soeur ?x ?u))
)

(defrule enfant_unique
	 (parent ?x ?y)
	 (not (frere_et_soeur ?x ?v))
=>
	(assert (enfant_unique ?x))
)
