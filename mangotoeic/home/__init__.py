print('=================home_init=================')
import logging
from flask import Blueprint
from flask_restful import Api
from mangotoeic.home.api import Home

home = Blueprint('home', __name__, url_prefix='/api')
api = Api(home)

api.add_resource(Home, '/api')

@home.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during home request. %s' % str(e))
    return 'An internal error occurred.', 500