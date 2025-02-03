from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from rdkit import Chem
from contextlib import contextmanager
import gzip
from io import StringIO

class SDFWriterError(Exception):
    """Custom exception for SDF writing errors"""
    pass

def process_mol_batch(batch_data: list[tuple], properties: list[str]) -> str:
    """Process a batch of molecules and their properties into SDF format strings"""
    sio = StringIO()
    writer = Chem.SDWriter(sio)
    
    for mol, row_dict in batch_data:
        try:
            if mol is None:
                continue
                
            # Create a copy of the molecule to avoid modifying the original
            mol_copy = Chem.Mol(mol)
            
            # Handle the molecule name from the row_dict
            if '_Name' in row_dict and row_dict['_Name'] is not None:
                mol_copy.SetProp('_Name', str(row_dict['_Name']))

            # Handle all other properties
            for prop, value in row_dict.items():
                if prop != '_Name' and value is not None:  # Skip None values and name property
                    if isinstance(value, float):
                        # Remove trailing zeros and ensure proper decimal point
                        s = f'{value:f}'.rstrip('0')
                        if s.endswith('.'):
                            s += '0'
                        mol_copy.SetProp(prop, s)
                    else:
                        mol_copy.SetProp(prop, str(value))
            
            writer.write(mol_copy)
            
        except Exception as e:
            print(f"Error processing molecule: {str(e)}")
            continue
    
    writer.close()
    return sio.getvalue()

class OptimizedSDFWriter:
    """High-performance SDF writer with batch processing capabilities"""
    
    def __init__(
        self,
        output_path: Path,
        molColName: str,
        idName: str | None = None,
        batch_size: int = 1000,
        n_cpus: int | None = None,
        properties: list[str] | None = None,
        compress: bool = False,
    ):
        self.output_path = output_path
        self.molColName = molColName
        self.idName = idName
        self.batch_size = batch_size
        self.n_cpus = n_cpus if n_cpus is not None else max(1, int(os.cpu_count() * 0.9))
        self.compress = compress
        self.properties = self._clean_properties(properties) if properties is not None else []
        
    def _clean_properties(self, properties: list[str]) -> list[str]:
        """Remove molColName and idName from properties list to avoid duplication"""
        exclude_cols = {self.molColName}
        if self.idName:
            exclude_cols.add(self.idName)
        return [prop for prop in properties if prop not in exclude_cols]
        
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame before processing"""
        if self.molColName not in df.columns:
            raise SDFWriterError(f"Molecule column '{self.molColName}' not found in DataFrame")
            
        if self.idName and self.idName not in df.columns:
            raise SDFWriterError(f"ID column '{self.idName}' not found in DataFrame")
            
        if not all(isinstance(mol, (Chem.Mol, type(None))) for mol in df[self.molColName]):
            raise SDFWriterError(f"Column '{self.molColName}' must contain only RDKit molecules")
            
        # Validate property columns exist
        missing_props = [prop for prop in self.properties if prop not in df.columns]
        if missing_props:
            raise SDFWriterError(f"Property columns not found in DataFrame: {missing_props}")
            
    @contextmanager
    def _get_output_file(self):
        """Context manager for handling output file with optional compression"""
        if self.compress:
            f = gzip.open(self.output_path, 'wt')
        else:
            f = open(self.output_path, 'w')
        try:
            yield f
        finally:
            f.close()
            
    def _prepare_batch_data(self, df: pd.DataFrame) -> list[list[tuple]]:
        """Prepare data in batches for parallel processing"""
        batch_data = []
        current_batch = []
        
        chunk_size = min(100000, len(df))
        for chunk_idx in range(0, len(df), chunk_size):
            chunk = df.iloc[chunk_idx:chunk_idx + chunk_size]
            
            for _, row in chunk.iterrows():
                mol = row[self.molColName]
                
                # Skip None molecules
                if mol is None:
                    continue
                    
                # Prepare properties dictionary
                props = {}
                
                # Handle ID column
                if self.idName:
                    props['_Name'] = str(row[self.idName])
                
                # Add selected DataFrame column properties
                for prop in self.properties:
                    props[prop] = row[prop]
                        
                current_batch.append((mol, props))
                
                if len(current_batch) >= self.batch_size:
                    batch_data.append(current_batch)
                    current_batch = []
                    
        if current_batch:
            batch_data.append(current_batch)
            
        return batch_data
        
    def write_sdf(self, df: pd.DataFrame) -> None:
        """
        Write DataFrame to SDF file with parallel processing
        
        Args:
            df: Input DataFrame containing molecules and properties
        """
        try:
            self._validate_dataframe(df)
            batch_data = self._prepare_batch_data(df)
            
            with self._get_output_file() as outfile:
                buffer = []
                buffer_size = 10 * 1024 * 1024  # 10MB buffer
                
                with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
                    futures = [
                        executor.submit(process_mol_batch, batch, self.properties)
                        for batch in batch_data
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            mol_block = future.result()
                            if mol_block:
                                buffer.append(mol_block)
                                
                                # Write buffer if it exceeds size threshold
                                if sum(len(b) for b in buffer) > buffer_size:
                                    outfile.write(''.join(buffer))
                                    buffer = []
                                    
                        except Exception as e:
                            print(f"Error processing batch: {str(e)}")
                            continue
                    
                    # Write any remaining buffer content
                    if buffer:
                        outfile.write(''.join(buffer))
                        
        except Exception as e:
            raise SDFWriterError(f"Error writing SDF file: {str(e)}")

def fast_write_sdf(
    df: pd.DataFrame,
    output_path: Path,
    molColName: str = 'Molecule',
    idName: str | None = None,
    properties: list[str] | None = None,
    batch_size: int = 1000,
    n_cpus: int | None = int(os.cpu_count()*0.9),
    compress: bool = False
) -> None:
    """
    High-performance parallel SDF writer with batch processing.
    
    Args:
        df: Input DataFrame containing molecules and properties
        output_path: Path to output SDF file
        molColName: Name of column containing RDKit molecules
        idName: Optional name of column to use as molecule names
        properties: Optional list of DataFrame column names to write as properties.
                   If None, no properties are written. Use list(df.columns) to write
                   all columns (molecule and ID columns will be handled appropriately).
        batch_size: Number of molecules to process in each batch
        n_cpus: Number of CPUs to use (default: 90% of available CPUs)
        compress: Whether to compress output as .gz file
    """
    writer = OptimizedSDFWriter(
        output_path=output_path,
        molColName=molColName,
        idName=idName,
        batch_size=batch_size,
        n_cpus=n_cpus,
        properties=properties,
        compress=compress
    )
    writer.write_sdf(df)
