#!/usr/bin/python
#-*- coding:utf-8 -*-
from flask import Flask
from flask_restful import reqparse, Api, Resource
from LDAModel import *

# 非监督学习，提前训练好模型，并完成序列化到文件
lda = LDAModel()
lda.sampling()

# Restful Api
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('query', required=True, help="query is None")
parser.add_argument('topK', type=int, default=8)

class CRS(Resource):
	def get(self):
		args = parser.parse_args()
		# apply LDAModel transform
		result = lda.transform(args['query'], args['topK'])
		return {'data': result}

api.add_resource(CRS, '/commodities/')

if __name__ == '__main__':
	app.run(debug=False)