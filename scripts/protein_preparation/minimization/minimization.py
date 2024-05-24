import sys
import os
import warnings
from pathlib import Path
import time
import numpy as np

from openmm import LangevinIntegrator
from openmm.app import (PME, ForceField, HBonds, Modeller, PDBFile, Simulation, )
from openmm.unit import femtoseconds, kelvin, nanometers, picoseconds
from pdbfixer import PDBFixer

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Fixing protein structure for creating a modeller object


def minimize_receptor(receptor: Path, solvent: bool = True) -> Path:
	try:
		fixer = PDBFixer(str(receptor))

		fixer.removeHeterogens()
		fixer.findNonstandardResidues()
		fixer.replaceNonstandardResidues()
		fixer.findMissingResidues()
		fixer.findMissingAtoms()
		fixer.addMissingAtoms()
		fixer.addMissingHydrogens()

		# Save the fixed PDB file
		printlog("Fixing receptor prior to minimization ...")
		receptor_fixed = receptor.with_name(receptor.stem + "_fixed.pdb")
		with open(receptor_fixed, "w") as f:
			PDBFile.writeFile(fixer.topology, fixer.positions, f)

		pdb = PDBFile(str(receptor_fixed))
		os.unlink(receptor_fixed)
	except Exception as e:
		printlog(f"Error fixing receptor prior to minimization: {e}")
		raise ValueError(f"Error fixing receptor prior to minimization: {e}")
	# Create a Modeller object
	try:
		forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
		modeller = Modeller(pdb.topology, pdb.positions)
		modeller.deleteWater()
		if solvent:
			printlog("Adding solvent to the receptor ...")
			modeller.addSolvent(forcefield, padding=1.0 * nanometers)
		else:
			pass
	except Exception as e:
		printlog(f"Error setting up Model prior to minimization: {e}")
		raise ValueError(f"Error setting up Model prior to minimization: {e}")
	# Create the system with a preliminary cutoff
	try:
		preliminary_cutoff = 1.0 * nanometers
		system = forcefield.createSystem(modeller.topology,
											nonbondedMethod=PME,
											nonbondedCutoff=preliminary_cutoff,
											constraints=HBonds)

		# Check box size and adjust cutoff
		box_vectors = system.getDefaultPeriodicBoxVectors()
		min_box_edge = min([np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in box_vectors])
		safe_cutoff = min(min_box_edge / 2, preliminary_cutoff)
		if safe_cutoff != preliminary_cutoff:
			# Re-create system with new cutoff
			system = forcefield.createSystem(modeller.topology,
												nonbondedMethod=PME,
												nonbondedCutoff=safe_cutoff,
												constraints=HBonds)
	except Exception as e:
		printlog(f"Error creating system prior to minimization: {e}")
		raise ValueError(f"Error creating system prior to minimization: {e}")
	try:
		# Set up the integrator
		integrator = LangevinIntegrator(300 * kelvin, 1 / picoseconds, 0.002 * femtoseconds)

		# Create the simulation object
		simulation = Simulation(modeller.topology, system, integrator)

		# Set the initial positions
		simulation.context.setPositions(modeller.positions)

		# Minimize the energy
		printlog("Minimizing receptor...")

		start_time = time.time()

		simulation.minimizeEnergy()

		end_time = time.time()
		time_taken = end_time - start_time

		print("Receptor minimization complete. Time taken: ", time_taken)

		# Get the minimized positions
		minimized_positions = simulation.context.getState(getPositions=True).getPositions()

		receptor_fixed_minimized = receptor.with_name(receptor.stem + "_minimized.pdb")
		# Save the minimized structure to a PDB file
		with open(receptor_fixed_minimized, "w") as f:
			PDBFile.writeFile(simulation.topology, minimized_positions, f)

		if solvent:
			fixer = PDBFixer(str(receptor_fixed_minimized))
			fixer.removeHeterogens(keepWater=False)
			PDBFile.writeFile(fixer.topology, fixer.positions, open(receptor_fixed_minimized, "w"))
	except Exception as e:
		printlog(f"Error minimizing receptor: {e}")
		raise ValueError(f"Error minimizing receptor: {e}")
	return receptor_fixed_minimized
