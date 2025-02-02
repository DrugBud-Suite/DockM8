from pathlib import Path
import os
from typing import NamedTuple
from collections.abc import Iterator
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from rdkit import Chem
from contextlib import contextmanager
from functools import partial
import array

class MoleculeData(NamedTuple):
    """Lightweight container for essential molecule data"""
    id: str
    properties: dict[str, str]
    mol_block: str  # Store molecular block instead of RDKit object for serialization

class SDFValidationError(Exception):
    """Custom exception for SDF validation errors"""
    pass

def process_batch_worker(
    batch_data: list[bytes],
    idName: str,
    required_props: set[str] | None = None
) -> list[MoleculeData | None]:
    """Process a batch of molecules efficiently"""
    results = []
    
    for mol_data in batch_data:
        try:
            # Use supplier without sanitization for initial read
            supplier = Chem.SDMolSupplier()
            supplier.SetData(mol_data.decode('utf-8'), sanitize=False)
            mol = next(supplier)
            
            if mol is None:
                continue

            # Get molecule name, ensuring it matches RDKit's behavior
            mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''

            # Extract properties, including name handling
            properties = {}
            if required_props:
                for prop in required_props:
                    if mol.HasProp(prop):
                        properties[prop] = mol.GetProp(prop)
            else:
                for prop in mol.GetPropNames():
                    if prop != '_Name':  # Skip _Name as it's handled separately
                        properties[prop] = mol.GetProp(prop)
            
            # Always include ID property
            properties[idName] = mol_name

            # Store molecular block instead of full RDKit object
            mol_block = Chem.MolToMolBlock(mol)
            
            results.append(MoleculeData(
                id=mol_name,
                properties=properties,
                mol_block=mol_block
            ))
            
        except Exception:
            continue
            
    return results

def process_mol_batch(batch_data: list[tuple]) -> list[tuple]:
    """Process a batch of molecule data, combining property handling and molecule creation"""
    results = []
    for mol_block, props in batch_data:
        try:
            mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
            if mol is not None and 'ID' in props:
                # Set the molecule name from the ID property
                mol.SetProp('_Name', str(props['ID']))
            results.append((mol, props))
        except Exception:
            results.append((None, props))
    return results

class OptimizedSDFLoader:
    """High-performance SDF loader with batch processing"""
    
    def __init__(
        self,
        sdf_path: Path,
        molColName: str,
        idName: str,
        batch_size: int = 10000,
        n_cpus: int | None = None,
        required_props: set[str] | None = None
    ):
        self.sdf_path = sdf_path
        self.molColName = molColName
        self.idName = idName
        self.batch_size = batch_size
        self.n_cpus = n_cpus if n_cpus is not None else max(1, int(os.cpu_count() * 0.9))
        self.required_props = required_props

    def _validate_input_file(self) -> None:
        """Validate SDF file before processing"""
        if not self.sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {self.sdf_path}")
        
        if self.sdf_path.stat().st_size == 0:
            raise SDFValidationError("SDF file is empty")
        
    @contextmanager
    def _mapped_file(self):
        """Memory-mapped file context manager"""
        mm = None
        try:
            with open(self.sdf_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                yield mm
        finally:
            if mm and not mm.closed:
                mm.close()

    def _extract_molecules(self, mm: mmap.mmap) -> Iterator[bytes]:
        """Efficiently extract molecule blocks"""
        current_molecule = array.array('B')
        
        for line in iter(mm.readline, b''):
            current_molecule.frombytes(line)
            
            if line.strip() == b'$$$$':
                yield current_molecule.tobytes()
                current_molecule = array.array('B')

    def _process_batches(
        self,
        molecule_data: list[bytes],
        total_molecules: int
    ) -> list[MoleculeData]:
        """Process molecules in optimized batches"""
        results = []
        batches = [
            molecule_data[i:i + self.batch_size]
            for i in range(0, len(molecule_data), self.batch_size)
        ]
        
        process_func = partial(
            process_batch_worker,
            idName=self.idName,
            required_props=self.required_props
        )
        
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = [
                executor.submit(process_func, batch)
                for batch in batches
            ]
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception:
                    continue
                    
        return results

    def _parallel_create_molecules(self, mol_blocks: list[str]) -> list[Chem.Mol | None]:
        """Create RDKit molecules in parallel"""
        def process_mol_batch(mol_block_batch: list[str]) -> list[Chem.Mol | None]:
            return [
                Chem.MolFromMolBlock(block, sanitize=True)
                for block in mol_block_batch
            ]

        # Split into batches
        batches = [
            mol_blocks[i:i + self.batch_size]
            for i in range(0, len(mol_blocks), self.batch_size)
        ]
        
        mol_objects = []
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            
            # Submit all batches
            for mol_block_batch in batches:
                futures.append(executor.submit(process_mol_batch, mol_block_batch))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    mol_objects.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    mol_objects.extend([None] * self.batch_size)
                    
        return mol_objects

    def _optimize_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame column types using pandas' modern type system.
        Handles both numeric and string columns intelligently while preserving precision.
        """
        exclude_cols = {self.idName, self.molColName}
        
        # Create a copy to avoid modifying the original during optimization
        df_optimized = df.copy()
        
        # Remove molecule column temporarily as it's not compatible with convert_dtypes
        mol_column = df_optimized[self.molColName]
        df_optimized = df_optimized.drop(columns=[self.molColName])
        
        # Use pandas' modern type inference system
        df_optimized = df_optimized.convert_dtypes(
            convert_integer=True,
            convert_boolean=True,
            convert_floating=True,
            convert_string=True
        )
        
        # Restore molecule column
        df_optimized[self.molColName] = mol_column
        
        return df_optimized

    def _create_dataframe(self, data: list[MoleculeData]) -> pd.DataFrame:
        """Create optimized DataFrame with robust type handling for all columns"""
        if not data:
            return pd.DataFrame()
        
        batch_data = [(d.mol_block, {self.idName: d.id, **d.properties}) for d in data]
        batches = [
            batch_data[i:i + self.batch_size]
            for i in range(0, len(batch_data), self.batch_size)
        ]
        
        results = []
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = [executor.submit(process_mol_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    continue
        
        mols, props = zip(*results, strict=False)
        
        # Create initial DataFrame
        df = pd.DataFrame(props)
        df[self.molColName] = mols
        
        df = self._optimize_column_types(df)
        
        return df

    def load(self) -> pd.DataFrame:
        """Load SDF file into DataFrame"""
        try:
            self._validate_input_file()
            
            with self._mapped_file() as mm:
                molecule_data = list(self._extract_molecules(mm))
                
                total_molecules = len(molecule_data)
                batches = [
                    molecule_data[i:i + self.batch_size]
                    for i in range(0, total_molecules, self.batch_size)
                ]
                
                results = []
                with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
                    futures = [
                        executor.submit(
                            process_batch_worker,
                            batch,
                            self.idName,
                            self.required_props
                        )
                        for batch in batches
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            results.extend(batch_results)
                        except Exception:
                            continue
                            
                return self._create_dataframe(results)
                
        except Exception as e:
            print(f"Error during SDF loading: {str(e)}")
            return pd.DataFrame()

def fast_load_sdf(
    sdf_path: Path,
    molColName: str,
    idName: str,
    batch_size: int = 100,
    n_cpus: int | None = int(os.cpu_count()*0.9),
    required_props: set[str] | None = None
) -> pd.DataFrame:
    """
    High-performance parallel SDF loader with batch processing.
    
    Args:
        sdf_path: Path to the SDF file
        molColName: Name of the molecule column
        idName: Name of the ID column
        batch_size: Number of molecules to process in each batch (default: 10000)
        n_cpus: Number of CPUs to use (default: 90% of available CPUs)
        required_props: Optional set of property names to extract
        
    Returns:
        DataFrame containing the processed SDF data
    """
    loader = OptimizedSDFLoader(
        sdf_path,
        molColName,
        idName,
        batch_size,
        n_cpus,
        required_props
    )
    return loader.load()
