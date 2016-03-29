from twisted.web.static import File
from klein import run, route

# @route('/slate/', branch=True)
# def static(request):
#     return File("./")

@route('/static/', branch=True)
def static(request):
    return File("./static")
    
@route('/slate')
def home(request):
    return File('index.html')

run("localhost", 2020)