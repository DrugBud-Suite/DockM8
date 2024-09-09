from base import DBObject


class Protomer(DBObject):

	def __init__(self, config, molregno, smiles):
		super().__init__(config)
		self.molregno = molregno
		self.smiles = smiles
		self.protomer_id = None

	def register(self):
		query = "INSERT INTO protomers (molregno, smiles) VALUES (%s, %s) RETURNING protomer_id"
		result = self._execute_query(query, (self.molregno, self.smiles))
		self.protomer_id = result[0][0]

	def query(self):
		query = "SELECT * FROM protomers WHERE molregno = %s AND smiles = %s"
		return self._execute_query(query, (self.molregno, self.smiles))

	def retrieve(self):
		query = "SELECT * FROM protomers WHERE protomer_id = %s"
		return self._execute_query(query, (self.protomer_id, ))

	def generate_conformers(self):
		# Similar to Molecule.generate_conformers()
		pass
