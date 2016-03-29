"""
This file is part of the flask+d3 Hello World project.
"""
import json
import flask
from flask import request
import math
from Awale import *


application = flask.Flask(__name__)


@application.route("/resolution", methods=['POST','GET'])

def resolution():
	idDataSet = str(request.json)
	result = main(idDataSet)
	return str(result)

@application.route("/affiche", methods=['POST','GET'])

def affiche():
	data = str(request.json)
	
	result = getInfosGrid(data)

	return str(result)


@application.route("/deleteAllTheGrids", methods=['POST','GET'])
def deleteAllTheGrids():
	print 'Deleting all'
	result = deleteGrids()
	return str(result)







@application.route("/", methods=['GET','POST'])
@application.route("/pywele", methods=['GET','POST'])
def awele():
	# if request.method == 'POST':

	# 	rowsMin = int(request.form['rowsMin'])
	# 	rowsMax = int(request.form['rowsMax'])
	# 	obsMin = int(request.form['obsMin'])
	# 	obsMax = int(request.form['obsMax'])
		
					
	# 	nbGrid = int(request.form['nbGrid'])
		
	# 	print "trying to generate grid "
	# 	t = createGrids(rowsMin,rowsMax,obsMin,obsMax,nbGrid)
	# 	t = math.floor(t*1000)/1000
	# 	l = getListOfGrids()
	# 	return flask.render_template("generate.html",text="Grid Generated in "+str(t)+" seconds",grids = l)
	
	return flask.render_template("index.html")
			
	# else:
	# 	print "NOPOST"
	# 	return flask.render_template("generate.html",grids = l,j=0)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()