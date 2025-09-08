# src/ontology_config.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


@dataclass
class OntologyConfig:
    name: str
    data_dir: str
    sapbert_dir: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    schema_url: Optional[str] = None
    documentation_url: Optional[str] = None
    paper_url: Optional[str] = None

    def __post_init__(self):
        self.base_path = Path(__file__).resolve().parent.parent
        self.data_path = self.base_path / self.data_dir

    @property
    def description_file(self):
        return str(self.data_path / f"{self.name}_short_description.json")

    @property
    def embedding_file(self):
        return str(self.data_path / f"all_{self.name}_mapped_vectors.json")

    @property
    def qualified_predicate_file(self):
        return str(self.data_path / "qualified_predicate_mappings.json")

    @property
    def sapbert_paths(self):
        if not self.sapbert_dir:
            return None, None, None, None

        sapbert_path = self.base_path / self.sapbert_dir
        return (
            sapbert_path / "data" / "mappings_dataset.txt",
            sapbert_path / "data" / "mappings_dictionary.txt",
            sapbert_path / "data" / "embedding_mappings.npy",
            sapbert_path / "model"
        )

    def get_details(self) -> Dict:
        """Get detailed information about this ontology"""
        details = {
            "name": self.display_name or self.name.title(),
            "description": self.description or f"{self.name} ontology"
        }

        if self.source_url:
            details["source"] = self.source_url
        if self.schema_url:
            details["schema"] = self.schema_url
        if self.documentation_url:
            details["documentation"] = self.documentation_url
        if self.paper_url:
            details["paper"] = self.paper_url

        return details


# Available ontologies with metadata
ONTOLOGIES = {
    "biolink": OntologyConfig(
        name="biolink",
        data_dir="biolink_data",
        sapbert_dir="Biolink_SapBert",
        display_name="Biolink Model",
        description="A high level datamodel of biological entities and associations",
        source_url="https://github.com/biolink/biolink-model",
        paper_url="https://ascpt.onlinelibrary.wiley.com/doi/10.1111/cts.13302",
        schema_url="https://github.com/biolink/biolink-model/blob/master/biolink-model.yaml",
        documentation_url="https://biolink.github.io/biolink-model/"
    ),
    "chemprot": OntologyConfig(
        name="chemprot",
        data_dir="chemprot_data",
        sapbert_dir="Chemprot_SapBert",
        display_name="ChemProt",
        description="Chemical-protein interaction corpus for biomedical relation extraction",
        source_url="https://huggingface.co/datasets/bigbio/chemprot",
        paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC3013776/",
        documentation_url="https://orbit.dtu.dk/files/122379096/ChemProt_3.0.pdf"
    )

}

# "new_ontology_goes_here": OntologyConfig(
#     name="new_ontology",
#     data_dir="new_data",
#     display_name="My New Ontology",
#     description="Description here",
#     source_url="https://...",
#     # etc.
# )

# Current ontology from environment variable
CURRENT_ONTOLOGY = os.getenv("ONTOLOGY", "biolink")


def get_current_config() -> OntologyConfig:
    """Get the current ontology configuration"""
    if CURRENT_ONTOLOGY not in ONTOLOGIES:
        raise ValueError(f"Unknown ontology: {CURRENT_ONTOLOGY}. Available: {list(ONTOLOGIES.keys())}")
    return ONTOLOGIES[CURRENT_ONTOLOGY]


def set_ontology(ontology_name: str) -> OntologyConfig:
    """Set the current ontology"""
    global CURRENT_ONTOLOGY
    if ontology_name not in ONTOLOGIES:
        raise ValueError(f"Unknown ontology: {ontology_name}. Available: {list(ONTOLOGIES.keys())}")
    CURRENT_ONTOLOGY = ontology_name
    return ONTOLOGIES[ontology_name]


def list_ontologies():
    """List available ontologies"""
    return list(ONTOLOGIES.keys())


def get_ontology_details():
    """Get detailed information about all ontologies"""
    return {name: config.get_details() for name, config in ONTOLOGIES.items()}