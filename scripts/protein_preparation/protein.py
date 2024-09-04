import io
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openmm
import pandas as pd
import requests
from Bio.PDB import PDBIO, PDBParser, Structure
from openmm import LangevinIntegrator, System
from openmm.app import PDBFile
from openmm.unit import femtoseconds, kelvin, nanometers, picoseconds
from pdbfixer import PDBFixer

from scripts.utilities.logging import printlog


class ProteinSource(Enum):
	PDB = "PDB"
	ALPHAFOLD = "AlphaFold"
	LOCAL = "Local File"


class Protein:

	def __init__(self, identifier: str, w_dir: Optional[Path] = None):
		"""
		Initializes a Protein object.

		Args:
			identifier (str): The identifier of the protein.
			w_dir (Optional[Path], optional): The working directory path. Defaults to None.
		"""
		self.identifier: str = identifier
		self.source: ProteinSource = self._determine_source(identifier)
		self.structure: Optional[Structure] = None
		self.pdb_file: Optional[Path] = None
		self.chains: List[str] = []
		self.resolution: Optional[float] = None
		self.sequence: Optional[str] = None
		self.is_fixed: bool = False
		self.is_prepared: bool = False
		self.edia_scores: Dict[str, float] = {}

		if self.source == ProteinSource.LOCAL:
			self.pdb_file = Path(identifier).resolve()
			self._parse_structure()
		else:
			if w_dir:
				self.w_dir = Path(w_dir)
			else:
				self.w_dir = self.create_temp_dir()
			self.fetch_structure(self.w_dir)

		self._parse_structure()

	@staticmethod
	def _determine_source(identifier: str) -> ProteinSource:
		"""
		Determine the source of the protein based on the identifier.

		Args:
			identifier (str): The protein identifier.

		Returns:
			ProteinSource: The determined source of the protein.

		Raises:
			ValueError: If the source cannot be determined.
		"""
		if len(identifier) == 4 and identifier.isalnum():
			return ProteinSource.PDB
		elif len(identifier) == 6 and identifier.isalnum():
			return ProteinSource.ALPHAFOLD
		elif Path(identifier).is_file():
			return ProteinSource.LOCAL
		else:
			raise ValueError(f"Unable to determine source for identifier: {identifier}")

	def fetch_structure(self, output_dir: Path) -> Path:
		"""
		Fetch the protein structure from the specified source and save it to a file.

		Args:
			output_dir (Path): The directory to save the fetched structure.

		Returns:
			Path: The path to the saved PDB file.

		Raises:
			ValueError: If the protein source is not supported or if the structure cannot be fetched.
		"""
		try:
			output_dir.mkdir(parents=True, exist_ok=True)

			if self.source == ProteinSource.PDB:
				url = f"https://files.rcsb.org/download/{self.identifier}.pdb"
				response = requests.get(url)
				response.raise_for_status()
				self.pdb_file = output_dir / f"{self.identifier}.pdb"
				with open(self.pdb_file, "w") as file:
					file.write(response.text)
				printlog(f"PDB file {self.pdb_file} downloaded successfully.")

			elif self.source == ProteinSource.ALPHAFOLD:
				url = f"https://alphafold.ebi.ac.uk/api/prediction/{self.identifier}"
				response = requests.get(url)
				response.raise_for_status()
				data = response.json()
				if data:
					pdb_url = data[0]["pdbUrl"]
					pdb_response = requests.get(pdb_url)
					pdb_response.raise_for_status()
					self.pdb_file = output_dir / f"{self.identifier}.pdb"
					with open(self.pdb_file, "wb") as file:
						file.write(pdb_response.content)
					printlog(f"AlphaFold structure downloaded and saved to: {self.pdb_file}")
				else:
					raise ValueError(f"No data available for UniProt code: {self.identifier}")

			else:
				raise ValueError(f"Unsupported protein source: {self.source}")

			self._parse_structure()
			return self.pdb_file
		except requests.RequestException as e:
			printlog(f"Error fetching structure: {e}")
			raise
		except Exception as e:
			printlog(f"Unexpected error in fetch_structure: {e}")
			raise

	def _parse_structure(self):
		"""
		Parse the PDB file and update the protein's attributes.

		Raises:
			Exception: If there's an error parsing the structure.
		"""
		try:
			parser = PDBParser()
			self.structure = parser.get_structure(self.identifier, self.pdb_file)
			self.chains = [chain.id for chain in self.structure.get_chains()]
			self.sequence = "".join(residue.resname for residue in self.structure.get_residues())
		except Exception as e:
			printlog(f"Error parsing structure: {e}")
			raise

	def fix_structure(self,
						fix_nonstandard_residues: bool = True,
						fix_missing_residues: bool = True,
						add_missing_hydrogens_pH: float = 7.0,
						remove_hetero: bool = True,
						remove_water: bool = True) -> None:
		"""
		Fix various issues in the protein structure.

		Args:
			fix_nonstandard_residues (bool, optional): Whether to fix nonstandard residues. Defaults to True.
			fix_missing_residues (bool, optional): Whether to fix missing residues. Defaults to True.
			add_missing_hydrogens_pH (float, optional): pH at which to add missing hydrogens. Defaults to 7.0.
			remove_hetero (bool, optional): Whether to remove heteroatoms. Defaults to True.
			remove_water (bool, optional): Whether to remove water molecules. Defaults to True.

		Raises:
			ValueError: If no PDB file is available.
			Exception: If there's an error fixing the structure.
		"""
		try:
			if self.pdb_file is None:
				raise ValueError("No PDB file available. Fetch or load a structure first.")

			fixer = PDBFixer(filename=str(self.pdb_file))

			if fix_nonstandard_residues:
				fixer.findNonstandardResidues()
				fixer.replaceNonstandardResidues()

			if fix_missing_residues:
				fixer.findMissingResidues()
				fixer.findMissingAtoms()
				fixer.addMissingAtoms()

			if add_missing_hydrogens_pH is not None:
				fixer.addMissingHydrogens(add_missing_hydrogens_pH)

			if remove_hetero and remove_water:
				fixer.removeHeterogens(keepWater=False)
			elif remove_hetero and not remove_water:
				fixer.removeHeterogens(keepWater=True)

			fixed_pdb = self.pdb_file.with_name(f"{self.pdb_file.stem}_fixed.pdb")
			PDBFile.writeFile(fixer.topology, fixer.positions, open(fixed_pdb, "w"))

			self.pdb_file = fixed_pdb
			self.is_fixed = True
			self._parse_structure()
			printlog(f"Structure fixed and saved to {self.pdb_file}")
		except Exception as e:
			printlog(f"Error fixing structure: {e}")
			raise

	def protonate(self, method: str = "protoss") -> None:
		"""
		Protonate the protein structure using the specified method.

		Args:
			method (str, optional): The protonation method to use. Defaults to "protoss".

		Raises:
			ValueError: If the specified method is not supported or if no PDB file is available.
			Exception: If there's an error during protonation.
		"""
		try:
			if method.lower() != "protoss":
				raise ValueError("Only 'protoss' method is currently supported for protonation.")

			if self.pdb_file is None:
				raise ValueError("No PDB file available. Fetch or load a structure first.")

			PROTEINS_PLUS_URL = "https://proteins.plus/api/v2/"
			PROTOSS = f"{PROTEINS_PLUS_URL}protoss/"
			PROTOSS_JOBS = f"{PROTEINS_PLUS_URL}protoss/jobs/"

			with open(self.pdb_file) as upload_file:
				query = {"protein_file": upload_file}
				job_submission = requests.post(PROTOSS, files=query).json()

			job_id = job_submission["job_id"]
			status = "pending"

			while status in ["pending", "running"]:
				response = requests.get(f"{PROTOSS_JOBS}{job_id}/").json()
				status = response["status"]
				if status == "failed":
					raise Exception("Protoss job failed")
				elif status == "completed":
					break

			protossed_protein = requests.get(
				f"{PROTEINS_PLUS_URL}molecule_handler/proteins/{response['output_protein']}/").json()
			protein_file = io.StringIO(protossed_protein["file_string"])

			parser = PDBParser()
			structure = parser.get_structure(self.identifier, protein_file)

			protonated_pdb = self.pdb_file.with_name(f"{self.pdb_file.stem}_protonated.pdb")
			writer = PDBIO()
			writer.set_structure(structure)
			writer.save(str(protonated_pdb))

			self.pdb_file = protonated_pdb
			self._parse_structure()
			printlog(f"Structure protonated and saved to {self.pdb_file}")
		except Exception as e:
			printlog(f"Error protonating structure: {e}")
			raise

	def minimize(self, solvent: bool = True) -> None:
		"""
		Perform energy minimization on the protein structure.

		Args:
			solvent (bool, optional): Whether to include solvent in the minimization. Defaults to True.

		Raises:
			ValueError: If no PDB file is available.
			Exception: If there's an error during minimization.
		"""
		try:
			if self.pdb_file is None:
				raise ValueError("No PDB file available. Fetch or load a structure first.")

			if not self.is_fixed:
				self.fix_structure()
				#self.protonate()

			pdb = PDBFile(str(self.pdb_file))
			forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

			modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
			modeller.deleteWater()

			if solvent:
				modeller.addSolvent(forcefield, padding=1.0 * nanometers)

			system = forcefield.createSystem(modeller.topology,
												nonbondedMethod=openmm.app.PME,
												nonbondedCutoff=1.0 * nanometers,
												constraints=openmm.app.HBonds)

			integrator = openmm.LangevinIntegrator(300 * kelvin, 1 / picoseconds, 0.002 * picoseconds)
			simulation = openmm.app.Simulation(modeller.topology, system, integrator)
			simulation.context.setPositions(modeller.positions)

			printlog("Minimizing energy...")
			simulation.minimizeEnergy()

			state = simulation.context.getState(getPositions=True, getEnergy=True)
			positions = state.getPositions()

			minimized_pdb = self.pdb_file.with_name(f"{self.pdb_file.stem}_minimized.pdb")
			with open(minimized_pdb, 'w') as f:
				PDBFile.writeFile(simulation.topology, positions, f)

			if solvent:
				fixer = PDBFixer(str(minimized_pdb))
				fixer.removeHeterogens(keepWater=False)
				PDBFile.writeFile(fixer.topology, fixer.positions, open(minimized_pdb, "w"))

			self.pdb_file = minimized_pdb
			try:
				parser = PDBParser()
				structure = parser.get_structure(self.identifier, minimized_pdb)
				# If parsing succeeds, update the pdb_file
				self.pdb_file = minimized_pdb
				self._parse_structure()
				printlog(f"Structure minimized and saved to {self.pdb_file}")
			except Exception as parse_error:
				printlog(f"Error parsing minimized structure: {parse_error}")
				printlog(f"Structure minimized and saved to {self.pdb_file}")
		except Exception as e:
			printlog(f"Error minimizing structure: {e}")
			raise

	def prepare_protein(self,
						fix_structure: bool = True,
						protonate: bool = True,
						minimize: bool = False,
						**kwargs) -> None:
		"""
		Prepare the protein structure by fixing, protonating, and minimizing.

		Args:
			fix_structure (bool, optional): Whether to fix the structure. Defaults to True.
			protonate (bool, optional): Whether to protonate the structure. Defaults to True.
			minimize (bool, optional): Whether to minimize the structure. Defaults to True.
			**kwargs: Additional keyword arguments for the individual preparation steps.

		Raises:
			Exception: If there's an error during protein preparation.
		"""
		try:
			if fix_structure:
				self.fix_structure(**kwargs)
			if protonate:
				self.protonate(**kwargs)
			if self.source == ProteinSource.ALPHAFOLD:
				minimize = True
			if minimize:
				self.minimize(**kwargs)

			# Rename the final PDB file to end with _prepared.pdb
			prepared_pdb = self.pdb_file.with_name(f"{self.pdb_file.stem.split('_')[0]}_prepared.pdb")
			shutil.move(self.pdb_file, prepared_pdb)
			self.pdb_file = prepared_pdb

			self.is_prepared = True
			printlog(f"Protein preparation completed. Final structure saved as {self.pdb_file}")
		except Exception as e:
			printlog(f"Error in protein preparation: {e}")
			raise

	def analyze_structure(self) -> Dict[str, Any]:
		"""
		Analyze the protein structure and return basic information.

		Returns:
			Dict[str, Any]: A dictionary containing the analysis results.

		Raises:
			ValueError: If no structure is available.
			Exception: If there's an error during analysis.
		"""
		try:
			if self.structure is None:
				raise ValueError("No structure available. Fetch or load a structure first.")

			analysis = {
				"chains": self.chains,
				"residues": sum(1 for _ in self.structure.get_residues()),
				"atoms": sum(1 for _ in self.structure.get_atoms()), }
			return analysis
		except Exception as e:
			printlog(f"Error analyzing structure: {e}")
			raise

	def get_best_chain(self) -> str:
		"""
		Get the best chain based on EDIA scores. If EDIA scores are not available, it runs the EDIA analysis first.

		Returns:
			str: The ID of the chain with the highest average EDIA score.

		Raises:
			ValueError: If the protein source is not PDB or if there's an error getting the best chain.
		"""
		try:
			if self.source != ProteinSource.PDB:
				raise ValueError("EDIA analysis is only available for structures from the PDB.")

			if not self.edia_scores:
				self._run_edia_analysis()

			return max(self.edia_scores, key=self.edia_scores.get)
		except Exception as e:
			printlog(f"Error getting best chain: {e}")
			raise

	def _run_edia_analysis(self) -> None:
		"""
		Run EDIA analysis on the protein structure and extract the best chain.

		Raises:
			ValueError: If no PDB file is available or if the protein source is not PDB.
			Exception: If there's an error during EDIA analysis or chain extraction.
		"""
		try:
			if self.pdb_file is None:
				raise ValueError("No PDB file available. Fetch or load a structure first.")

			if self.source != ProteinSource.PDB:
				raise ValueError("EDIA analysis is only available for structures from the PDB.")

			EDIA_API_URL = "https://proteins.plus/api/edia_rest"
			headers = {"Content-Type": "application/json", "Accept": "application/json"}
			data = {"edia": {"pdbCode": self.identifier}}

			response = requests.post(EDIA_API_URL, json=data, headers=headers)
			response.raise_for_status()
			job_url = response.json()["location"]
			job_response = requests.get(job_url)
			structure_scores_url = job_response.json()["structure_scores"]
			scores_response = requests.get(structure_scores_url)
			scores_response.raise_for_status()

			scores_df = pd.read_csv(io.StringIO(scores_response.text))
			self.edia_scores = scores_df.groupby("Chain")["EDIAm"].mean().to_dict()
			printlog("EDIA analysis completed.")

			# Extract the best chain
			best_chain = max(self.edia_scores, key=self.edia_scores.get)
			best_chain_file = self._extract_chain(self.pdb_file, best_chain)

			if best_chain_file:
				self.pdb_file = best_chain_file
				self._parse_structure()
				printlog(f"Best chain (Chain {best_chain}) extracted and saved to {self.pdb_file}")
			else:
				printlog(f"Failed to extract best chain (Chain {best_chain})")

		except requests.RequestException as e:
			printlog(
				f"Error in EDIA API request: {e}. If the structure is an NMR or Cryo-EM structure, EDIA cannot be run. Please check your structure before trying to run EDIA again."
			)
			raise
		except Exception as e:
			printlog(
				f"Error running EDIA analysis or extracting best chain: {e}. If the structure is an NMR or Cryo-EM structure, EDIA cannot be run. Please check your structure before trying to run EDIA again."
			)
			raise

	def _extract_chain(self, pdb_file: Path, chain_id: str) -> Optional[Path]:
		"""
		Extract a specific chain from a PDB file.

		Args:
			pdb_file (Path): The path to the input PDB file.
			chain_id (str): The ID of the chain to extract.

		Returns:
			Optional[Path]: The path to the output PDB file containing only the specified chain, or None if extraction fails.
		"""
		try:
			parser = PDBParser()
			structure = parser.get_structure("structure", pdb_file)

			pdbio = PDBIO()

			for model in structure:
				for chain in model:
					if chain.get_id() == chain_id:
						pdbio.set_structure(chain)
						output_file = pdb_file.parent / f"{pdb_file.stem}_{chain_id}.pdb"
						pdbio.save(str(output_file))
						return output_file

			printlog(f"Chain {chain_id} not found in the structure.")
			return None

		except FileNotFoundError:
			printlog(f"Error: PDB file '{pdb_file}' not found.")
		except Exception as e:
			printlog(f"Error in extracting chain {chain_id} from {pdb_file}: {e}")

		return None

	def to_pdb(self, output_path: Path) -> Path:
		"""
		Save the current protein structure to a PDB file.

		Args:
			output_path (Path): The path where the PDB file should be saved.

		Returns:
			Path: The path of the saved PDB file.

		Raises:
			ValueError: If no structure is available.
			Exception: If there's an error saving the structure to PDB.
		"""
		try:
			if self.structure is None:
				raise ValueError("No structure available. Fetch or load a structure first.")

			io = PDBIO()
			io.set_structure(self.structure)
			io.save(str(output_path))
			printlog(f"Structure saved to {output_path}")
			return output_path
		except Exception as e:
			printlog(f"Error saving structure to PDB: {e}")
			raise

	def get_sequence(self) -> str:
		"""
		Get the amino acid sequence of the protein.

		Returns:
			str: The amino acid sequence of the protein.

		Raises:
			ValueError: If no sequence is available.
			Exception: If there's an error getting the sequence.
		"""
		try:
			if self.sequence is None:
				raise ValueError("No sequence available. Fetch or load a structure first.")
			return self.sequence
		except Exception as e:
			printlog(f"Error getting sequence: {e}")
			raise

	def __str__(self) -> str:
		"""
		Return a string representation of the Protein object.

		Returns:
			str: A string representation of the Protein object.
		"""
		return f"Protein {self.identifier} from {self.source.value}"

	def __repr__(self) -> str:
		"""
		Return a string representation of the Protein object that can be used to recreate the object.

		Returns:
			str: A string representation of the Protein object.
		"""
		return f"Protein(identifier='{self.identifier}', source={self.source})"

	def create_temp_dir(self) -> Path:
		"""
		Creates a temporary directory for the docking function.

		Returns:
			Path: The path to the temporary directory.
		"""
		os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
		temp_dir = Path(Path.home() / "dockm8_temp_files" / f"dockm8_{self.identifier}_{os.getpid()}")
		temp_dir.mkdir(parents=True, exist_ok=True)
		return temp_dir

	@staticmethod
	def remove_temp_dir(temp_dir: Path):
		"""
		Removes the temporary directory.

		Args:
			temp_dir (Path): The path to the temporary directory.
		"""
		shutil.rmtree(str(temp_dir), ignore_errors=True)
