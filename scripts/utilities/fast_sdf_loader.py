from pathlib import Path
import os
from typing import NamedTuple
from collections.abc import Iterator
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from contextlib import contextmanager
from functools import partial
import array

BATCH_SIZE = 10000  # Process molecules in batches for better efficiency

class MoleculeData(NamedTuple):
    """Lightweight container for essential molecule data"""
    id: str
    properties: dict[str, str]
    mol_block: str  # Store molecular block instead of RDKit object for serialization

def process_batch_worker(
    batch_data: list[bytes],
    id_name: str,
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
            properties[id_name] = mol_name

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
        mol_col_name: str,
        id_name: str,
        n_cpus: int | None = None,
        required_props: set[str] | None = None
    ):
        self.sdf_path = sdf_path
        self.mol_col_name = mol_col_name
        self.id_name = id_name
        self.n_cpus = n_cpus if n_cpus is not None else max(1, int(os.cpu_count() * 0.9))
        self.required_props = required_props
        
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
            molecule_data[i:i + BATCH_SIZE]
            for i in range(0, len(molecule_data), BATCH_SIZE)
        ]
        
        process_func = partial(
            process_batch_worker,
            id_name=self.id_name,
            required_props=self.required_props
        )
        
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = [
                executor.submit(process_func, batch)
                for batch in batches
            ]
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing molecule batches"
            ):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception:
                    continue
                    
        return results

    def _parallel_create_molecules(self, mol_blocks: list[str], batch_size: int = 5000) -> list[Chem.Mol | None]:
        """Create RDKit molecules in parallel"""
        def process_mol_batch(mol_block_batch: list[str]) -> list[Chem.Mol | None]:
            return [
                Chem.MolFromMolBlock(block, sanitize=True)
                for block in mol_block_batch
            ]

        # Split into batches
        batches = [
            mol_blocks[i:i + batch_size]
            for i in range(0, len(mol_blocks), batch_size)
        ]
        
        mol_objects = []
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = []
            
            # Submit all batches
            for mol_block_batch in batches:
                futures.append(executor.submit(process_mol_batch, mol_block_batch))
            
            # Collect results
            for future in tqdm(
                as_completed(futures),
                total=len(batches),
                desc="Converting to RDKit molecules"
            ):
                try:
                    batch_results = future.result()
                    mol_objects.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    mol_objects.extend([None] * batch_size)
                    
        return mol_objects

    def _optimize_numeric_columns(self, df: pd.DataFrame) -> None:
        """Optimize numeric columns in-place using parallel processing"""
        exclude_cols = {self.id_name, self.mol_col_name, "ID", "Pose ID", "SMILES"}
        numeric_cols = [col for col in df.columns if col not in exclude_cols]
        
        def convert_column(col: str) -> pd.Series:
            return pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            future_to_col = {
                executor.submit(convert_column, col): col
                for col in numeric_cols
            }
            
            for future in as_completed(future_to_col):
                col = future_to_col[future]
                try:
                    df[col] = future.result()
                except Exception:
                    pass

    def _create_dataframe(self, data: list[MoleculeData]) -> pd.DataFrame:
        """Create optimized DataFrame with combined property and molecule processing"""
        if not data:
            return pd.DataFrame()
        
        print("Preparing data for parallel processing...")
        # Prepare combined data for processing
        batch_data = [(d.mol_block, {self.id_name: d.id, **d.properties}) for d in data]
        batch_size = 10000  # Increased batch size for better performance
        batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
        
        print("Processing molecules and properties...")
        results = []
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            futures = [executor.submit(process_mol_batch, batch) for batch in batches]
            
            for future in tqdm(as_completed(futures), total=len(batches),
                             desc="Processing batches"):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
                    continue
        
        print("Creating final DataFrame...")
        # Separate molecules and properties
        mols, props = zip(*results, strict=False)
        
        # Create DataFrame from properties
        df = pd.DataFrame(props)
        df[self.mol_col_name] = mols
        
        print("Optimizing numeric columns...")
        # Convert numeric columns efficiently
        exclude_cols = {self.id_name, self.mol_col_name, "ID", "Pose ID", "SMILES"}
        numeric_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Process numeric columns in larger chunks
        chunk_size = 1000
        for col in numeric_cols:
            chunks = [df[col][i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            converted_chunks = []
            for chunk in chunks:
                converted_chunks.append(pd.to_numeric(chunk, errors='coerce', downcast='float'))
            df[col] = pd.concat(converted_chunks)
        
        return df

    def load(self) -> pd.DataFrame:
        """Main loading method with optimized batch processing"""
        try:
            # Count molecules efficiently
            with open(self.sdf_path, 'rb') as f:
                total_molecules = f.read().count(b'$$$$')
            
            with self._mapped_file() as mm:
                # Extract molecules using memory mapping
                molecule_data = list(tqdm(
                    self._extract_molecules(mm),
                    total=total_molecules,
                    desc="Reading molecules"
                ))
                
                # Process in optimized batches
                processed_data = self._process_batches(molecule_data, total_molecules)
                
                return self._create_dataframe(processed_data)
                
        except Exception as e:
            print(f"Error during SDF loading: {str(e)}")
            return pd.DataFrame()

def fast_load_sdf(
    sdf_path: Path,
    mol_col_name: str,
    id_name: str,
    n_cpus: int | None = None,
    required_props: set[str] | None = None
) -> pd.DataFrame:
    """
    High-performance parallel SDF loader with batch processing.
    
    Args:
        sdf_path: Path to the SDF file
        mol_col_name: Name of the molecule column
        id_name: Name of the ID column
        n_cpus: Number of CPUs to use
        required_props: Optional set of property names to extract
        
    Returns:
        DataFrame containing the processed SDF data
    """
    loader = OptimizedSDFLoader(sdf_path, mol_col_name, id_name, n_cpus, required_props)
    return loader.load()
