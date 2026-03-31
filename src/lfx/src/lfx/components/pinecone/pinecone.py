"""Pinecone Vector Store component for Langflow.

Dual-mode component that extends LCVectorStoreComponent:
  - **Upsert mode** – When no search query is provided, the component reads
    ingest_data (DataFrame / list[Data]) where the ``text`` column holds the
    text to embed and every other column is stored as Pinecone metadata.
  - **Search mode** – When a search query is provided, it embeds the query
    and performs a similarity search, returning the top-k results.

Uses the official Pinecone Python SDK (``pinecone``) for direct control over
upsert payloads and metadata handling.
"""

from __future__ import annotations

import json
# import logging
import sys
import uuid
from datetime import datetime
from typing import Any

import numpy as np

# File-based debug logger — guaranteed to work regardless of threading
# _LOG_FILE = r"D:\Work\langflow\langflow\pinecone_debug.log"


# def _debug_log(msg: str) -> None:
#     """Write a debug line to both stderr and a log file."""
#     timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
#     line = f"[PINECONE {timestamp}] {msg}"
#     # Write to stderr (more reliable than stdout in threaded context)
#     print(line, file=sys.stderr, flush=True)
#     # Also write to a file
#     try:
#         with open(_LOG_FILE, "a", encoding="utf-8") as f:
#             f.write(line + "\n")
#     except Exception:
#         pass

# _debug_log("=== pinecone.py MODULE LOADED ===")

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.field_typing import VectorStore
from lfx.io import (
    HandleInput,
    IntInput,
    Output,
    QueryInput,
    SecretStrInput,
    StrInput,
)
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame

# Internal Langflow keys to skip — only the ones that are large nested
# objects (risk exceeding Pinecone's 40 KB limit) or purely internal framework
# plumbing with no document relevance.
# _SKIP_METADATA_KEYS = frozenset(
#     {
#         "properties",       # large nested dict (UI state, colours, etc.)
#         "content_blocks",   # list of complex content block objects
#         "text_key",         # internal Langflow routing key
#         "default_value",    # internal Langflow default
#     }
# )


class PineconeVectorStoreComponent(LCVectorStoreComponent):
    """Pinecone component – upserts table data or searches an index.

    Extends ``LCVectorStoreComponent`` so Langflow's runtime correctly
    discovers and invokes the component via ``search_documents()``.

    *  Connect **Data / DataFrame** to the *Ingest Data* handle and leave
       *Search Query* empty → the component **upserts** the rows.
    *  Fill in a **Search Query** → the component **searches** the index.
    """

    display_name = "Pinecone"
    description = "Pinecone Vector Store with search capabilities"
    name = "Pinecone"
    icon = "Pinecone"

    inputs = [
        SecretStrInput(
            name="pinecone_api_key",
            display_name="Pinecone API Key",
            required=True,
            info="Your Pinecone API key.",
        ),
        StrInput(
            name="index_name",
            display_name="Index Name",
            required=True,
            info="Name of the Pinecone index to upsert into / search.",
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            info="Pinecone namespace (leave blank for the default namespace).",
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Connect an embedding model to generate vectors.",
        ),
        HandleInput(
            name="ingest_data",
            display_name="Ingest Data",
            input_types=["Data", "DataFrame"],
            is_list=True,
            info=(
                "Table data to upsert.  The 'text' column is embedded; "
                "all other columns are stored as metadata."
            ),
        ),
        QueryInput(
            name="search_query",
            display_name="Search Query",
            info="When provided the component performs a similarity search instead of upserting.",
            tool_mode=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="How many results to return from a search.",
            value=4,
            advanced=True,
        ),
        IntInput(
            name="upsert_batch_size",
            display_name="Upsert Batch Size",
            info="Number of vectors to upsert per batch (max 1000).",
            value=100,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Search Results",
            name="search_results",
            method="search_documents",
        ),
        Output(
            display_name="Table",
            name="dataframe",
            method="as_dataframe",
        ),
    ]

    # ------------------------------------------------------------------
    # Required by LCVectorStoreComponent
    # ------------------------------------------------------------------

    @check_cached_vector_store
    def build_vector_store(self) -> VectorStore:
        """Not used directly — we bypass langchain VectorStore and use the
        Pinecone SDK directly via ``search_documents()`` and ``_upsert()``.

        Returns a lightweight sentinel so the base class doesn't error.
        """
        # _debug_log("build_vector_store() called")
        # Return None — we never actually use a langchain VectorStore object.
        # Our search_documents() override handles everything.
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_pinecone_index(self):
        """Return a Pinecone Index object using the official SDK."""
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            msg = (
                "The 'pinecone' package is required.  "
                "Install it with:  pip install pinecone"
            )
            raise ImportError(msg) from exc

        pc = Pinecone(api_key=self.pinecone_api_key)
        return pc.Index(self.index_name)

    @staticmethod
    def _force_float32(vec: list) -> list[float]:
        """Ensure every element is a plain Python ``float`` (float32-safe).

        Pinecone rejects float64 values — numpy float32 cast avoids this.
        """
        return [float(np.float32(x)) for x in vec]

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> str | int | float | bool | None:
        """Coerce a single metadata value into a Pinecone-safe type.

        Pinecone accepts: str, int, float, bool, list[str].
        Everything else is JSON-serialised to a string.  ``None`` is dropped.
        """
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, list):
            return json.dumps(value)
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)

    def _flatten_item(self, item: Any) -> list[dict[str, Any]]:
        """Convert a single ingest item into a list of plain dicts.

        Handles: Data, DataFrame, dict, and nested lists of any depth.
        """
        if isinstance(item, DataFrame):
            return [row.copy() for row in item.to_dict(orient="records")]
        if isinstance(item, Data):
            return [item.data.copy()]
        if isinstance(item, dict):
            return [item.copy()]
        if isinstance(item, list):
            # Recurse into nested lists (e.g. [[Data, Data, ...]])
            results: list[dict[str, Any]] = []
            for sub in item:
                results.extend(self._flatten_item(sub))
            return results
        return []

    def _prepare_rows(self) -> list[dict[str, Any]]:
        """Convert ``ingest_data`` into a flat list of dicts.

        Handles nested lists (Langflow delivers ``[[Data, …]]``), DataFrames,
        and single Data objects.
        """
        raw = self.ingest_data
        if not raw:
            return []

        if not isinstance(raw, list):
            raw = [raw]

        rows: list[dict[str, Any]] = []
        for item in raw:
            rows.extend(self._flatten_item(item))

        self.log(f"_prepare_rows: flattened {len(rows)} rows from ingest_data.")
        return rows

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _upsert(self) -> list[Data]:
        """Embed and upsert rows into Pinecone.

        For each row the ``text`` column is embedded.  Every other column is
        stored as flat metadata (internal Langflow keys are skipped;
        nested dicts/lists are JSON-serialised).

        Uses the Pinecone SDK ``index.upsert()`` with the format::

            index.upsert(
                vectors=[
                    {"id": "...", "values": [...], "metadata": {...}},
                    ...
                ],
                namespace="..."
            )
        """
        # _debug_log("_upsert() called")
        # _debug_log(f"_upsert: ingest_data type={type(self.ingest_data)}, truthy={bool(self.ingest_data)}")
        # if isinstance(self.ingest_data, list):
            # _debug_log(f"_upsert: ingest_data len={len(self.ingest_data)}")
            # if self.ingest_data and isinstance(self.ingest_data[0], list):
                # _debug_log(f"_upsert: ingest_data[0] is a list, len={len(self.ingest_data[0])}")
                # if self.ingest_data[0]:
                #     _debug_log(f"_upsert: ingest_data[0][0] type={type(self.ingest_data[0][0])}")

        rows = self._prepare_rows()
        # _debug_log(f"_upsert: got {len(rows)} rows after _prepare_rows()")
        if not rows:
            # _debug_log("_upsert: NO ROWS — returning empty")
            self.log("_upsert: No data to upsert — ingest_data was empty or could not be parsed.")
            return [
                Data(
                    data={
                        "text": "No data to upsert. Check that ingest_data is connected.",
                        "status": "empty",
                    }
                )
            ]

        if not self.embedding:
            msg = "An Embedding Model must be connected to upsert data."
            raise ValueError(msg)

        index = self._get_pinecone_index()
        namespace = self.namespace or ""
        batch_size = min(self.upsert_batch_size or 100, 1000)  # Pinecone max is 1000

        texts: list[str] = []
        metadata_list: list[dict] = []

        for row in rows:
            text = str(row.pop("text", ""))
            texts.append(text)

            # Build clean metadata — skip internal keys and None values
            flat_meta: dict[str, Any] = {}
            for k, v in row.items():
                # if k in _SKIP_METADATA_KEYS:
                #     continue
                sanitized = self._sanitize_metadata_value(v)
                if sanitized is not None:
                    flat_meta[k] = sanitized

            # Store the original text in metadata so we can return it on search
            flat_meta["text"] = text
            metadata_list.append(flat_meta)

        # _debug_log(f"_upsert: Embedding {len(texts)} texts...")
        self.log(f"_upsert: Embedding {len(texts)} texts...")

        # Embed all texts at once
        try:
            vectors = self.embedding.embed_documents(texts)
            dim = len(vectors[0]) if vectors else "?"
            # _debug_log(f"_upsert: Generated {len(vectors)} embeddings (dim={dim})")
            self.log(f"_upsert: Generated {len(vectors)} embeddings (dim={dim}).")
        except Exception as e:
            # _debug_log(f"_upsert: EMBEDDING ERROR: {e}")
            raise

        # Upsert in batches — Pinecone format: list of dicts with id, values, metadata
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i : i + batch_size]
            batch_meta = metadata_list[i : i + batch_size]

            upsert_payload = []
            for vec, meta in zip(batch_vectors, batch_meta):
                vec_id = str(uuid.uuid4())
                upsert_payload.append(
                    {
                        "id": vec_id,
                        "values": self._force_float32(vec),
                        "metadata": meta,
                    }
                )

            # _debug_log(f"_upsert: Sending batch to Pinecone: {len(upsert_payload)} vectors, namespace='{namespace}'")
            try:
                resp = index.upsert(vectors=upsert_payload, namespace=namespace)
                total_upserted += len(upsert_payload)
                # _debug_log(f"_upsert: Pinecone response: {resp}")
                self.log(
                    f"_upsert: Batch {i // batch_size + 1} — "
                    f"sent {len(upsert_payload)} vectors (total: {total_upserted}). "
                    f"Response: {resp}"
                )
            except Exception as e:
                # _debug_log(f"_upsert: PINECONE UPSERT ERROR: {e}")
                import traceback
                traceback.print_exc()
                raise

        self.log(
            f"✅ Upserted {total_upserted} vectors into "
            f"'{self.index_name}' (namespace='{namespace}')."
        )

        result = [
            Data(
                data={
                    "text": (
                        f"Successfully upserted {total_upserted} vectors "
                        f"into index '{self.index_name}'."
                    ),
                    "status": "success",
                    "upserted_count": total_upserted,
                    "index": self.index_name,
                    "namespace": namespace,
                }
            )
        ]
        self.status = result
        return result

    def _search(self) -> list[Data]:
        """Embed the search query and return the top-k results from Pinecone."""
        if not self.embedding:
            msg = "An Embedding Model must be connected to search."
            raise ValueError(msg)

        query_text: str = self.search_query.strip()
        if not query_text:
            return []

        index = self._get_pinecone_index()
        namespace = self.namespace or ""
        k = self.number_of_results or 4

        self.log(
            f"_search: Querying '{self.index_name}' "
            f"(namespace='{namespace}') for: {query_text[:100]}..."
        )

        query_vector = self.embedding.embed_query(query_text)
        query_vector = self._force_float32(query_vector)

        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            namespace=namespace,
        )

        data_list: list[Data] = []
        for match in results.get("matches", []):
            meta = dict(match.get("metadata", {}))
            text = meta.pop("text", "")
            score = match.get("score", 0.0)
            data_list.append(
                Data(
                    data={
                        "text": text,
                        "score": score,
                        "id": match.get("id", ""),
                        **meta,
                    }
                )
            )

        self.log(f"_search: Found {len(data_list)} results.")
        self.status = data_list
        return data_list

    # ------------------------------------------------------------------
    # Output methods (override LCVectorStoreComponent)
    # ------------------------------------------------------------------

    def search_documents(self) -> list[Data]:
        """Override base class ``search_documents``.

        Routes to upsert or search depending on whether a search query
        is provided.  This is the method Langflow's runtime calls.
        """
        # _debug_log(">>> search_documents() OVERRIDE CALLED <<<")
        search_query = getattr(self, "search_query", None)
        # _debug_log(f"search_query = {repr(search_query)}, type = {type(search_query)}")
        if search_query and isinstance(search_query, str) and search_query.strip():
            # _debug_log("Routing to _search()")
            self.log("search_documents: search_query is set → performing search.")
            return self._search()
        # _debug_log("Routing to _upsert()")
        self.log("search_documents: no search_query → performing upsert.")
        return self._upsert()

    def as_dataframe(self) -> DataFrame:
        """Return search results (or upsert confirmation) as a DataFrame."""
        return DataFrame(self.search_documents())
