"""Call GraphRAG's local_search/global_search Python API.

graphrag.api is explicitly marked "under development" (backwards
compatibility not guaranteed) by GraphRAG itself. Loading the parquet
output tables into the DataFrames these functions require is normally
done by a private CLI helper (graphrag.cli.query._resolve_output_files),
so this module reimplements that loading step with the public building
blocks it uses internally (graphrag_storage + DataReader). Re-check this
file when upgrading the graphrag dependency.

graphrag.config.load_config also chdirs into the config file's directory
by default (set_cwd=True) to resolve relative paths. That side effect is
undone immediately after loading, before any DataFrame is read.
"""

import asyncio
import os
from pathlib import Path

import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.data_model.data_reader import DataReader
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

_GRAPHRAG_ROOT = Path(__file__).parent.parent / "graphrag"


def _load_graphrag_config():
    original_cwd = Path.cwd()
    try:
        return load_config(root_dir=_GRAPHRAG_ROOT)
    finally:
        os.chdir(original_cwd)


def _load_output_tables(
    config, output_list: list[str], optional_list: list[str] | None = None
) -> dict:
    storage_obj = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)

    dataframe_dict = {
        name: asyncio.run(getattr(reader, name)()) for name in output_list
    }
    for name in optional_list or []:
        has_table = asyncio.run(table_provider.has(name))
        dataframe_dict[name] = (
            asyncio.run(getattr(reader, name)()) if has_table else None
        )
    return dataframe_dict


def local_search(
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """Answer a query using GraphRAG's local search."""
    config = _load_graphrag_config()
    data = _load_output_tables(
        config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        optional_list=["covariates"],
    )
    response, _context = asyncio.run(
        api.local_search(
            config=config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            text_units=data["text_units"],
            relationships=data["relationships"],
            covariates=data["covariates"],
            community_level=community_level,
            response_type=response_type,
            query=query,
        )
    )
    return response


def global_search(
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> str:
    """Answer a query using GraphRAG's global search."""
    config = _load_graphrag_config()
    data = _load_output_tables(
        config,
        output_list=["entities", "communities", "community_reports"],
    )
    response, _context = asyncio.run(
        api.global_search(
            config=config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            community_level=community_level,
            dynamic_community_selection=False,
            response_type=response_type,
            query=query,
        )
    )
    return response
