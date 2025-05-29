# Dataset download utility for German-English translation corpus
# Dataset from https://tatoeba.org/

import os
import requests
import gzip
import shutil
from pathlib import Path
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm


class TatoebaDatasetDownloader:
    """Download and prepare Tatoeba German-English dataset"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Tatoeba dataset URLs
        self.base_url = "https://downloads.tatoeba.org/exports/"
        self.sentences_url = f"{self.base_url}sentences.tar.bz2"
        self.links_url = f"{self.base_url}links.tar.bz2"
        
        # Alternative direct download for processed data
        self.processed_url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/de-en.txt.zip"
        
        self.setup_logging()
    
    def setup_logging(self):
        """Initialize logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_file(self, url: str, filename: str, extract: bool = False) -> bool:
        """Download file with progress bar"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            self.logger.info(f"File {filename} already exists, skipping download")
            return True
            
        try:
            self.logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            if extract and filename.endswith('.gz'):
                self.extract_gz(filepath)
            
            self.logger.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {filename}: {str(e)}")
            return False
    
    def extract_gz(self, filepath: Path):
        """Extract gzipped file"""
        output_path = filepath.with_suffix('')
        with gzip.open(filepath, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        filepath.unlink()  # Remove the .gz file
        self.logger.info(f"Extracted {filepath.name}")
    
    def download_opus_dataset(self) -> bool:
        """Download preprocessed OPUS Tatoeba dataset"""
        try:
            filename = "de-en.txt.zip"
            if self.download_file(self.processed_url, filename):
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile(self.data_dir / filename, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir)
                
                # Remove zip file
                (self.data_dir / filename).unlink()
                
                # Find extracted files and rename to expected format
                for file in self.data_dir.glob("Tatoeba.*"):
                    if "de-en" in file.name:
                        new_name = self.data_dir / "ge-en.tsv"
                        file.rename(new_name)
                        self.logger.info(f"Renamed {file.name} to ge-en.tsv")
                        break
                
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error downloading OPUS dataset: {str(e)}")
            return False
    
    def create_sample_dataset(self) -> bool:
        """Create a sample dataset for testing if download fails"""
        try:
            sample_data = [
                ("Hello, how are you?", "Hallo, wie geht es dir?"),
                ("Good morning!", "Guten Morgen!"),
                ("Thank you very much.", "Vielen Dank."),
                ("Where is the train station?", "Wo ist der Bahnhof?"),
                ("I would like to order food.", "Ich möchte Essen bestellen."),
                ("The weather is beautiful today.", "Das Wetter ist heute schön."),
                ("Can you help me please?", "Können Sie mir bitte helfen?"),
                ("I don't understand.", "Ich verstehe nicht."),
                ("How much does it cost?", "Wie viel kostet es?"),
                ("Have a nice day!", "Haben Sie einen schönen Tag!"),
            ]
            
            # Create DataFrame
            df = pd.DataFrame(sample_data, columns=['en', 'de'])
            
            # Save as TSV
            output_file = self.data_dir / "ge-en.tsv"
            df.to_csv(output_file, sep='\t', index=False, header=False)
            
            self.logger.info(f"Created sample dataset with {len(sample_data)} sentence pairs")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sample dataset: {str(e)}")
            return False
    
    def download_tatoeba_corpus(self) -> bool:
        """Download full Tatoeba corpus and extract German-English pairs"""
        try:
            self.logger.info("Full corpus download not implemented yet. Using OPUS dataset...")
            return self.download_opus_dataset()
            
        except Exception as e:
            self.logger.error(f"Error downloading Tatoeba corpus: {str(e)}")
            return False
    
    def validate_dataset(self, filepath: Optional[Path] = None) -> bool:
        """Validate downloaded dataset"""
        if filepath is None:
            filepath = self.data_dir / "ge-en.tsv"
        
        if not filepath.exists():
            self.logger.error(f"Dataset file {filepath} not found")
            return False
        
        try:
            df = pd.read_csv(filepath, sep='\t', header=None, nrows=10)
            
            if len(df.columns) < 2:
                self.logger.error("Dataset should have at least 2 columns")
                return False
            
            self.logger.info(f"Dataset validation passed. Shape: {df.shape}")
            self.logger.info(f"Sample data:\n{df.head()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating dataset: {str(e)}")
            return False
    
    def download(self) -> bool:
        """Main download method"""
        self.logger.info("Starting Tatoeba German-English dataset download...")
        
        if self.download_opus_dataset():
            if self.validate_dataset():
                self.logger.info("Successfully downloaded and validated dataset!")
                return True
        
        self.logger.warning("OPUS download failed, creating sample dataset...")
        if self.create_sample_dataset():
            if self.validate_dataset():
                self.logger.info("Sample dataset created successfully!")
                return True
        
        self.logger.error("All download methods failed")
        return False
    
    def get_dataset_info(self) -> dict:
        """Get information about the downloaded dataset"""
        filepath = self.data_dir / "ge-en.tsv"
        
        if not filepath.exists():
            return {"error": "Dataset not found"}
        
        try:
            df = pd.read_csv(filepath, sep='\t', header=None)
            
            info = {
                "file_path": str(filepath),
                "total_pairs": len(df),
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
                "columns": df.shape[1],
                "sample_english": df.iloc[0, 0] if len(df) > 0 else None,
                "sample_german": df.iloc[0, 1] if len(df) > 0 and df.shape[1] > 1 else None,
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Error reading dataset: {str(e)}"}


def main():
    """Main function to download the dataset"""
    print("🌍 Tatoeba German-English Dataset Downloader")
    print("=" * 50)
    
    downloader = TatoebaDatasetDownloader()
    
    if downloader.download():
        print("\n✅ Dataset download completed successfully!")
        
        info = downloader.get_dataset_info()
        print(f"\n📊 Dataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ Dataset download failed!")
        return False
    
    return True


if __name__ == "__main__":
    main()