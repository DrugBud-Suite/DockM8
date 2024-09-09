from base import DBObject
import json


class PocketFinder:

	@staticmethod
	def find_pocket(method, protein, **kwargs):
		# Implement pocket finding logic
		pocket_data = "PLACEHOLDER"
		return Pocket(protein.config, protein.protein_id, pocket_data)


class Pocket(DBObject):

	def __init__(self, config, protein_id, pocket_data):
		super().__init__(config)
		self.protein_id = protein_id
		self.pocket_data = pocket_data
		self.pocket_id = None

	def register(self):
		query = "INSERT INTO pockets (protein_id, pocket_data) VALUES (%s, %s) RETURNING pocket_id"
		result = self._execute_query(query, (self.protein_id, json.dumps(self.pocket_data)))
		self.pocket_id = result[0][0]

	def query(self):
		query = "SELECT * FROM pockets WHERE protein_id = %s"
		return self._execute_query(query, (self.protein_id, ))

	def retrieve(self):
		query = "SELECT * FROM pockets WHERE pocket_id = %s"
		return self._execute_query(query, (self.pocket_id, ))
