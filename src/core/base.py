import lwreg
from lwreg import utils


class DBObject:

	def __init__(self, config):
		self.config = config
		self.conn = lwreg.utils._connect(config)

	def _execute_query(self, query, params=None):
		with self.conn.cursor() as cursor:
			cursor.execute(query, params)
			self.conn.commit()
			return cursor.fetchall()

	def register(self):
		raise NotImplementedError

	def query(self):
		raise NotImplementedError

	def retrieve(self):
		raise NotImplementedError
