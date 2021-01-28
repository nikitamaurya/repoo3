# Flask library are imported here
import os
from flask import Flask


#Define app name
app = Flask(__name__)

#Set up key
app.config['SECRET_KEY']='1a858e5d5f93ac4338efe291b34f9d8c'
app.config['UPLOAD_FOLDER']= ""


#call home page
from frontend import routes
