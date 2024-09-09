from base import DBObject
import json


class DockingPose(DBObject):

	def __init__(self, config, conformer_id, docking_result):
		super().__init__(config)
		self.conformer_id = conformer_id
		self.docking_result = docking_result
		self.pose_id = None

	def register(self):
		query = "INSERT INTO docking_poses (conformer_id, docking_result) VALUES (%s, %s) RETURNING pose_id"
		result = self._execute_query(query, (self.conformer_id, json.dumps(self.docking_result)))
		self.pose_id = result[0][0]

	def query(self):
		query = "SELECT * FROM docking_poses WHERE conformer_id = %s"
		return self._execute_query(query, (self.conformer_id, ))

	def retrieve(self):
		query = "SELECT * FROM docking_poses WHERE pose_id = %s"
		return self._execute_query(query, (self.pose_id, ))
