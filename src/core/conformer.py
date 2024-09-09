from base import DBObject
import json
from docking_pose import DockingPose


class Conformer(DBObject):

	def __init__(self, config, molregno, conformer_data):
		super().__init__(config)
		self.molregno = molregno
		self.conformer_data = conformer_data
		self.conformer_id = None

	def register(self):
		query = "INSERT INTO conformers (molregno, conformer_data) VALUES (%s, %s) RETURNING conformer_id"
		result = self._execute_query(query, (self.molregno, json.dumps(self.conformer_data)))
		self.conformer_id = result[0][0]

	def query(self):
		query = "SELECT * FROM conformers WHERE molregno = %s"
		return self._execute_query(query, (self.molregno, ))

	def retrieve(self):
		query = "SELECT * FROM conformers WHERE conformer_id = %s"
		return self._execute_query(query, (self.conformer_id, ))

	def dock(self, protein, pocket):
		# Implement docking logic
		docking_result = "PLACEHOLDER"
		return DockingPose(self.config, self.conformer_id, docking_result)
