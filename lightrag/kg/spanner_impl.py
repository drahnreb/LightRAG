import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, final, Tuple, Set, Dict

# Ensure google-cloud-spanner and opentelemetry packages are installed
import pipmaster as pm

if not pm.is_installed("google-cloud-spanner"):
    pm.install("google-cloud-spanner>=3.53.0")
if not pm.is_installed("opentelemetry-api"):
    pm.install("opentelemetry-api>=1.0.0")
if not pm.is_installed("opentelemetry-sdk"):
    pm.install("opentelemetry-sdk>=1.0.0")
if not pm.is_installed("opentelemetry-exporter-gcp-trace"):
    pm.install("opentelemetry-exporter-gcp-trace>=1.0.0")
if not pm.is_installed("opentelemetry-instrumentation-grpc"):
    pm.install(
        "opentelemetry-instrumentation-grpc>=0.30b0"
    )  # Check for appropriate version


try:
    from google.cloud import spanner_v1
    from google.cloud.spanner_v1.database import Database
    from google.cloud.spanner_v1.pool import TransactionPingingPool
    from google.api_core import exceptions as google_exceptions

    # OpenTelemetry Imports
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    from opentelemetry.trace import Status, StatusCode

    # gRPC Instrumentation (optional, but good for Spanner)
    from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

    grpc_instrumentor = GrpcInstrumentorClient()
    grpc_instrumentor.instrument()  # Instrument gRPC globally once

except ImportError as e:
    print(f"Error importing required libraries: {e}. Please ensure installation.")
    raise


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..utils import logger


import configparser
from dotenv import load_dotenv

# --- Configuration Loading ---

load_dotenv(dotenv_path=".env", override=False)
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# --- Spanner Client Management ---


class SpannerClientManager:
    """Manages singleton Spanner Client, Database, Pool, and Tracer instances."""

    _instances: dict[str, Any] = {
        "client": None,
        "instance": None,
        "database": None,
        "pool": None,
        "tracer_provider": None,
        "tracer": None,
        "ref_count": 0,
    }
    _lock = asyncio.Lock()
    _db_checked = False  # Flag to check/create tables only once per process

    @staticmethod
    def get_config() -> dict[str, Any]:
        """Loads Spanner configuration from environment variables and config.ini."""
        config = configparser.ConfigParser()
        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_file_dir = os.getcwd()
        config_path = os.path.join(current_file_dir, "config.ini")

        if os.path.exists(config_path):
            config.read(config_path, "utf-8")
            logger.info(f"Loaded Spanner config from: {config_path}")
        else:
            logger.warning(
                f"config.ini not found at {config_path}, using environment variables or defaults."
            )

        spanner_config = {
            "project_id": os.environ.get(
                "SPANNER_PROJECT_ID", config.get("spanner", "project_id", fallback=None)
            ),
            "instance_id": os.environ.get(
                "SPANNER_INSTANCE_ID",
                config.get("spanner", "instance_id", fallback=None),
            ),
            "database_id": os.environ.get(
                "SPANNER_DATABASE_ID",
                config.get("spanner", "database_id", fallback=None),
            ),
            # Add pool configuration if needed (e.g., pool_size)
        }
        if not all(
            [
                spanner_config["project_id"],
                spanner_config["instance_id"],
                spanner_config["database_id"],
            ]
        ):
            raise ValueError(
                "Missing required Spanner configuration: project_id, instance_id, or database_id"
            )

        logger.info(
            f"Spanner Configuration resolved to: Project={spanner_config['project_id']}, Instance={spanner_config['instance_id']}, DB={spanner_config['database_id']}"
        )
        return spanner_config

    @classmethod
    async def get_client(
        cls,
    ) -> Tuple[Database, TransactionPingingPool, trace.Tracer]:
        """Gets or creates the singleton Spanner Database, Session Pool, and Tracer."""
        async with cls._lock:
            if (
                cls._instances["database"] is None
                or cls._instances["pool"] is None
                or cls._instances["tracer"] is None
            ):
                logger.info(
                    "Creating new Spanner Client, Database, Session Pool, and Tracer instances."
                )
                config = cls.get_config()
                project_id = config["project_id"]
                try:
                    # Setup OpenTelemetry Tracer
                    if cls._instances["tracer_provider"] is None:
                        tracer_provider = TracerProvider(sampler=ALWAYS_ON)
                        trace_exporter = CloudTraceSpanExporter(project_id=project_id)
                        tracer_provider.add_span_processor(
                            BatchSpanProcessor(trace_exporter)
                        )
                        trace.set_tracer_provider(tracer_provider)
                        cls._instances["tracer_provider"] = tracer_provider
                        logger.info(
                            f"Initialized OpenTelemetry TracerProvider for project {project_id}"
                        )
                    else:
                        tracer_provider = cls._instances["tracer_provider"]

                    # Retrieve a tracer instance
                    tracer = tracer_provider.get_tracer(
                        "lightrag.spanner_graph_storage"
                    )
                    cls._instances["tracer"] = tracer

                    # Setup Spanner Client
                    client = spanner_v1.SpannerAsyncClient()
                    instance = client.instance(config["instance_id"])
                    if not await instance.exists():
                        raise ValueError(
                            f"Spanner instance '{config['instance_id']}' not found in project '{project_id}'."
                        )

                    database = instance.database(config["database_id"])
                    if not await database.exists():
                        raise ValueError(
                            f"Spanner database '{config['database_id']}' not found in instance '{config['instance_id']}'."
                        )

                    # Create session pool
                    pool = TransactionPingingPool(
                        database=database, size=10, labels={"lightrag": "graph"}
                    )

                    cls._instances["client"] = client
                    cls._instances["instance"] = instance
                    cls._instances["database"] = database
                    cls._instances["pool"] = pool
                    cls._instances["ref_count"] = 0
                    cls._db_checked = False
                    logger.info(
                        "Spanner Client, Database, and Session Pool initialized."
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to initialize Spanner Client/DB/Pool/Tracer: {e}",
                        exc_info=True,
                    )
                    if cls._instances["pool"]:
                        cls._instances["pool"].close()
                    cls._instances = {
                        "client": None,
                        "instance": None,
                        "database": None,
                        "pool": None,
                        "tracer_provider": None,
                        "tracer": None,
                        "ref_count": 0,
                    }
                    raise

            cls._instances["ref_count"] += 1
            logger.debug(
                f"Spanner client reference count incremented to: {cls._instances['ref_count']}"
            )
            return (
                cls._instances["database"],
                cls._instances["pool"],
                cls._instances["tracer"],
            )

    @classmethod
    async def release_client(
        cls, db: Database, pool: TransactionPingingPool, tracer: trace.Tracer
    ):
        """Decrements reference count and closes pool/tracer if count reaches zero."""
        async with cls._lock:
            if (
                cls._instances["database"] is not None
                and db is cls._instances["database"]
                and cls._instances["pool"] is not None
                and pool is cls._instances["pool"]
                and cls._instances["tracer"] is not None
                and tracer is cls._instances["tracer"]
            ):
                cls._instances["ref_count"] -= 1
                logger.debug(
                    f"Spanner client reference count decremented to: {cls._instances['ref_count']}"
                )

                if cls._instances["ref_count"] <= 0:
                    logger.info(
                        "Reference count reached zero. Closing Spanner Session Pool and Tracer Provider..."
                    )
                    try:
                        if cls._instances["pool"]:
                            cls._instances["pool"].close()
                        if cls._instances["tracer_provider"]:
                            # Shutdown the tracer provider to flush spans
                            cls._instances["tracer_provider"].shutdown()

                        cls._instances = {
                            "client": None,
                            "instance": None,
                            "database": None,
                            "pool": None,
                            "tracer_provider": None,
                            "tracer": None,
                            "ref_count": 0,
                        }
                        cls._db_checked = False
                        logger.info(
                            "Spanner Session Pool and Tracer Provider closed/shutdown."
                        )
                    except Exception as e:
                        logger.error(
                            f"Error closing Spanner resources: {e}", exc_info=True
                        )
                        cls._instances = {
                            "client": None,
                            "instance": None,
                            "database": None,
                            "pool": None,
                            "tracer_provider": None,
                            "tracer": None,
                            "ref_count": 0,
                        }
                        cls._db_checked = False
            elif cls._instances["ref_count"] > 0:
                logger.debug(
                    f"Spanner client release called, but ref count is still {cls._instances['ref_count']}. Not closing."
                )
            else:
                logger.warning(
                    "Attempted to release Spanner client instances that don't match the managed singletons or were already released."
                )

    @classmethod
    def is_db_checked(cls) -> bool:
        return cls._db_checked

    @classmethod
    def set_db_checked(cls, status: bool):
        cls._db_checked = status


# --- Spanner Graph Storage Implementation ---


# Helper function to check for Spanner Aborted errors for retry
def _is_spanner_aborted(exception):
    return isinstance(exception, google_exceptions.Aborted)


@final
@dataclass
class SpannerGraphStorage(BaseGraphStorage):
    """
    Graph storage implementation using Google Cloud Spanner.

    Represents graph data using relational tables (Nodes, Edges) and
    stores properties as JSON. Includes OpenTelemetry tracing.
    """

    _db: Database | None = field(default=None, init=False, repr=False)
    _pool: TransactionPingingPool | None = field(default=None, init=False, repr=False)
    _tracer: trace.Tracer | None = field(
        default=None, init=False, repr=False
    )  # Add tracer instance variable
    _node_table: str = field(default="GraphNodes", init=False)
    _edge_table: str = field(default="GraphEdges", init=False)
    _entity_id_col: str = field(default="entityId", init=False)
    _properties_col: str = field(default="properties", init=False)
    _edge_source_col: str = field(default="sourceId", init=False)
    _edge_target_col: str = field(default="targetId", init=False)

    def __post_init__(self):
        super().__post_init__()  # Call super if BaseGraphStorage.__post_init__ exists
        # Define table names, possibly using namespace
        # For simplicity, using fixed names here, but namespace could be prefixed
        self._node_table = (
            f"GraphNodes_{self.namespace}" if self.namespace else "GraphNodes"
        )
        self._edge_table = (
            f"GraphEdges_{self.namespace}" if self.namespace else "GraphEdges"
        )
        logger.info(
            f"SpannerGraphStorage instance for namespace '{self.namespace}' created. Tables: {self._node_table}, {self._edge_table}"
        )

    async def initialize(self):
        """Initializes the Spanner connection, pool, tracer and ensures schema exists."""
        if not (self._db and self._pool and self._tracer):
            logger.info(
                f"SpannerGraphStorage ({self.namespace}) initializing client..."
            )
            try:
                (
                    self._db,
                    self._pool,
                    self._tracer,
                ) = await SpannerClientManager.get_client()
                if (
                    not await SpannerClientManager.is_db_checked()
                ):  # Made is_db_checked async
                    await self._ensure_schema()
                    await SpannerClientManager.set_db_checked(True)
                logger.info(
                    f"SpannerGraphStorage ({self.namespace}) client initialized."
                )
            except Exception as e:
                logger.error(
                    f"SpannerGraphStorage ({self.namespace}) init failed: {e}",
                    exc_info=True,
                )
                self._db = self._pool = self._tracer = None  # Ensure clean state
                raise
        else:
            logger.info(f"SpannerGraphStorage ({self.namespace}) already initialized.")

    async def finalize(self):
        """Releases the Spanner connection pool and tracer."""
        if self._db and self._pool and self._tracer:
            logger.info(f"SpannerGraphStorage ({self.namespace}) finalizing client...")
            await SpannerClientManager.release_client(
                self._db, self._pool, self._tracer
            )
            self._db = self._pool = self._tracer = None
            logger.info(f"SpannerGraphStorage ({self.namespace}) finalized.")
        else:
            logger.info(
                f"SpannerGraphStorage ({self.namespace}) already finalized or not initialized."
            )

    async def _ensure_schema(self):
        """Creates the necessary Spanner tables and indexes if they don't exist."""
        db = self._get_db()
        tracer = self._get_tracer()
        logger.info(
            f"Checking/Creating Spanner schema for tables: {self._node_table}, {self._edge_table}"
        )

        # DDL statements remain the same
        node_ddl = f"""
        CREATE TABLE {self._node_table} (
            {self._entity_id_col}   STRING(MAX) NOT NULL,
            {self._properties_col}  JSON
        ) PRIMARY KEY ({self._entity_id_col})
        """
        edge_ddl = f"""
        CREATE TABLE {self._edge_table} (
            {self._edge_source_col} STRING(MAX) NOT NULL,
            {self._edge_target_col} STRING(MAX) NOT NULL,
            {self._properties_col}  JSON,
            CONSTRAINT FK_EdgeSource FOREIGN KEY ({self._edge_source_col}) REFERENCES {self._node_table} ({self._entity_id_col}) ON DELETE CASCADE,
            CONSTRAINT FK_EdgeTarget FOREIGN KEY ({self._edge_target_col}) REFERENCES {self._node_table} ({self._entity_id_col}) ON DELETE CASCADE
        ) PRIMARY KEY ({self._edge_source_col}, {self._edge_target_col})
        """  # Added ON DELETE CASCADE for easier node deletion
        edge_source_index_ddl = f"CREATE INDEX Idx_EdgeSource ON {self._edge_table}({self._edge_source_col})"
        edge_target_index_ddl = f"CREATE INDEX Idx_EdgeTarget ON {self._edge_table}({self._edge_target_col})"

        statements = [node_ddl, edge_ddl, edge_source_index_ddl, edge_target_index_ddl]
        operations = []

        # Wrap the entire schema check/update in a span
        async with tracer.start_as_current_span("ensure_spanner_schema") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.name", db.database_id)
            span.set_attribute("db.statement_count", len(statements))

            all_succeeded = True
            for stmt in statements:
                stmt_short = stmt.strip().splitlines()[0]
                async with tracer.start_as_current_span(
                    f"update_ddl:{stmt_short}"
                ) as ddl_span:
                    ddl_span.set_attribute("db.statement", stmt)  # Log full DDL in span
                    try:
                        op = await db.update_ddl([stmt])
                        operations.append(op)
                        logger.info(f"Submitted DDL: {stmt_short}...")
                        ddl_span.set_status(Status(StatusCode.OK))
                    except google_exceptions.AlreadyExists:
                        logger.debug(f"Schema element already exists: {stmt_short}...")
                        ddl_span.set_status(
                            Status(StatusCode.OK, "Already Exists")
                        )  # OK status, add description
                    except Exception as e:
                        logger.error(
                            f"Error submitting DDL '{stmt_short}...': {e}",
                            exc_info=True,
                        )
                        ddl_span.set_status(Status(StatusCode.ERROR, str(e)))
                        ddl_span.record_exception(e)
                        all_succeeded = False
                        # Decide if we should raise or continue trying others
                        # For schema setup, maybe best to raise immediately
                        raise

            # Wait for submitted DDL operations to complete
            if operations:
                logger.info(
                    f"Waiting for {len(operations)} DDL operations to complete..."
                )
                async with tracer.start_as_current_span(
                    "wait_ddl_operations"
                ) as wait_span:
                    try:
                        results = await asyncio.gather(
                            *(op.result() for op in operations), return_exceptions=True
                        )
                        # Check results for any exceptions during DDL execution
                        for i, res in enumerate(results):
                            if isinstance(res, Exception):
                                logger.error(
                                    f"DDL operation {i} failed during execution: {res}"
                                )
                                wait_span.set_status(
                                    Status(
                                        StatusCode.ERROR,
                                        f"DDL operation {i} failed: {res}",
                                    )
                                )
                                wait_span.record_exception(res)
                                all_succeeded = False
                        if all_succeeded:
                            logger.info("DDL operations completed successfully.")
                            wait_span.set_status(Status(StatusCode.OK))
                        else:
                            logger.warning(
                                "Some DDL operations failed during execution."
                            )
                            # Status remains ERROR if any exception occurred

                    except Exception as e:
                        logger.error(
                            f"Error waiting for DDL operations: {e}", exc_info=True
                        )
                        wait_span.set_status(Status(StatusCode.ERROR, str(e)))
                        wait_span.record_exception(e)
                        all_succeeded = False

            # Set overall span status
            if not all_succeeded:
                span.set_status(Status(StatusCode.ERROR, "Schema update failed"))

    # --- Helper Methods ---

    def _get_db(self) -> Database:
        """Ensures the database object is available."""
        if not self._db:
            raise RuntimeError(
                "Spanner Database object not available. Ensure initialize() is called."
            )
        return self._db

    def _get_pool(self) -> TransactionPingingPool:
        """Ensures the session pool is available."""
        if not self._pool:
            raise RuntimeError(
                "Spanner Session Pool not available. Ensure initialize() is called."
            )
        return self._pool

    def _get_tracer(self) -> trace.Tracer:
        """Ensures the tracer object is available."""
        if not self._tracer:
            # This case should ideally not happen if initialize works correctly
            logger.error("Tracer accessed before initialization!")
            # Attempt to re-initialize - might be problematic in async context
            # A better approach is strict initialization enforcement
            raise RuntimeError("Tracer not available. Ensure initialize() is called.")
            # Alternatively, return a no-op tracer: return trace.get_tracer(__name__)
        return self._tracer

    # --- Core Graph Operations (Wrapped with Tracing) ---

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),  # Retry only on Aborted
    )
    async def has_node(self, node_id: str) -> bool:
        """Checks if a node with the given entity_id exists."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"SELECT EXISTS (SELECT 1 FROM {self._node_table} WHERE {self._entity_id_col} = @node_id)"
        params = {"node_id": node_id}
        param_types = {"node_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING)}

        async with tracer.start_as_current_span("has_node") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.node_id", node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        exists = row[0]
                        span.set_attribute("db.spanner.node_exists", exists)
                        span.set_status(Status(StatusCode.OK))
                        return exists
                # If loop finishes without returning (shouldn't happen for EXISTS)
                span.set_status(Status(StatusCode.ERROR, "Query returned no rows"))
                return False
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error checking node existence for '{node_id}': {e}", exc_info=True
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Checks if an edge exists between two nodes (in either direction)."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"""
            SELECT EXISTS (
                SELECT 1 FROM {self._edge_table}
                WHERE ({self._edge_source_col} = @node1 AND {self._edge_target_col} = @node2)
                   OR ({self._edge_source_col} = @node2 AND {self._edge_target_col} = @node1)
            )
        """
        params = {"node1": source_node_id, "node2": target_node_id}
        param_types = {
            "node1": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
            "node2": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
        }
        async with tracer.start_as_current_span("has_edge") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.source_node_id", source_node_id)
            span.set_attribute("db.spanner.target_node_id", target_node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        exists = row[0]
                        span.set_attribute("db.spanner.edge_exists", exists)
                        span.set_status(Status(StatusCode.OK))
                        return exists
                span.set_status(Status(StatusCode.ERROR, "Query returned no rows"))
                return False
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error checking edge existence between '{source_node_id}' and '{target_node_id}': {e}",
                    exc_info=True,
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Gets node properties (from JSON column) by entity_id."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"SELECT {self._properties_col} FROM {self._node_table} WHERE {self._entity_id_col} = @node_id"
        params = {"node_id": node_id}
        param_types = {"node_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING)}

        async with tracer.start_as_current_span("get_node") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.node_id", node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        props = row[0] if row[0] else {}
                        props[self._entity_id_col] = node_id
                        span.set_attribute("db.spanner.node_found", True)
                        span.set_status(Status(StatusCode.OK))
                        return props
                # If loop finishes, node was not found
                logger.debug(f"Node '{node_id}' not found in get_node.")
                span.set_attribute("db.spanner.node_found", False)
                span.set_status(Status(StatusCode.OK))  # OK status, just not found
                return None
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting node for '{node_id}': {e}", exc_info=True)
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def node_degree(self, node_id: str) -> int:
        """Gets the degree (number of relationships) of a node."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"SELECT COUNT(*) FROM {self._edge_table} WHERE {self._edge_source_col} = @node_id OR {self._edge_target_col} = @node_id"
        params = {"node_id": node_id}
        param_types = {"node_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING)}

        async with tracer.start_as_current_span("node_degree") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.node_id", node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        degree = row[0]
                        span.set_attribute("db.spanner.node_degree", degree)
                        span.set_status(Status(StatusCode.OK))
                        return degree
                # Should always return a row with count
                span.set_status(Status(StatusCode.ERROR, "Query returned no rows"))
                return 0
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if not _is_spanner_aborted(e):
                    logger.warning(
                        f"Error getting node degree for '{node_id}', returning 0: {e}",
                        exc_info=False,
                    )
                    return 0
                logger.error(
                    f"Error getting node degree for '{node_id}': {e}", exc_info=True
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
        reraise=True,
    )
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Calculates edge degree as sum of source and target node degrees."""
        tracer = self._get_tracer()
        async with tracer.start_as_current_span("edge_degree") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.spanner.source_node_id", src_id)
            span.set_attribute("db.spanner.target_node_id", tgt_id)
            try:
                # Get degrees of source and target nodes concurrently
                src_degree_task = self.node_degree(src_id)
                tgt_degree_task = self.node_degree(tgt_id)
                src_node_degree, tgt_node_degree = await asyncio.gather(
                    src_degree_task, tgt_degree_task
                )

                total_degree = src_node_degree + tgt_node_degree
                span.set_attribute("db.spanner.edge_degree", total_degree)
                span.set_status(Status(StatusCode.OK))
                return total_degree
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if not _is_spanner_aborted(e):
                    logger.warning(
                        f"Error getting edge degree for '{src_id}'->'{tgt_id}', returning 0: {e}",
                        exc_info=False,
                    )
                    return 0
                logger.error(
                    f"Error getting node degree for '{src_id}'->'{tgt_id}': {e}",
                    exc_info=True,
                )
                raise  # Reraise the exception

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, Any] | None:
        """Gets properties (from JSON column) of the first edge found between two nodes."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"""
            SELECT {self._properties_col} FROM {self._edge_table}
            WHERE ({self._edge_source_col} = @node1 AND {self._edge_target_col} = @node2)
               OR ({self._edge_source_col} = @node2 AND {self._edge_target_col} = @node1)
            LIMIT 1
        """
        params = {"node1": source_node_id, "node2": target_node_id}
        param_types = {
            "node1": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
            "node2": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
        }
        async with tracer.start_as_current_span("get_edge") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.source_node_id", source_node_id)
            span.set_attribute("db.spanner.target_node_id", target_node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        props = row[0] if row[0] else {}
                        span.set_attribute("db.spanner.edge_found", True)
                        span.set_status(Status(StatusCode.OK))
                        return props
                logger.debug(
                    f"Edge between '{source_node_id}' and '{target_node_id}' not found."
                )
                span.set_attribute("db.spanner.edge_found", False)
                span.set_status(Status(StatusCode.OK))
                return None
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error getting edge between '{source_node_id}' and '{target_node_id}': {e}",
                    exc_info=True,
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Gets all edges (source_id, target_id) connected to a node."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"""
            SELECT {self._edge_source_col}, {self._edge_target_col}
            FROM {self._edge_table}
            WHERE {self._edge_source_col} = @node_id OR {self._edge_target_col} = @node_id
        """
        params = {"node_id": source_node_id}
        param_types = {"node_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING)}
        edges = []
        async with tracer.start_as_current_span("get_node_edges") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.node_id", source_node_id)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        src, tgt = row[0], row[1]
                        if src == source_node_id:
                            edges.append((src, tgt))
                        else:
                            edges.append((tgt, src))
                span.set_attribute("db.spanner.edge_count", len(edges))
                span.set_status(Status(StatusCode.OK))
                return edges if edges else None
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error getting edges for node '{source_node_id}': {e}",
                    exc_info=True,
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, Any]) -> None:
        """Creates/updates a node using Spanner Mutation API."""
        db = self._get_db()
        tracer = self._get_tracer()
        if self._entity_id_col not in node_data:
            node_data[self._entity_id_col] = node_id
        elif node_data[self._entity_id_col] != node_id:
            logger.warning(
                f"Mismatch between node_id ('{node_id}') and '{self._entity_id_col}' in properties ('{node_data[self._entity_id_col]}'). Using node_id ('{node_id}')."
            )
            node_data[self._entity_id_col] = node_id

        properties_to_store = {
            k: v for k, v in node_data.items() if k != self._entity_id_col
        }

        async with tracer.start_as_current_span("upsert_node") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "insert_or_update")
            span.set_attribute("db.spanner.table", self._node_table)
            span.set_attribute("db.spanner.node_id", node_id)
            try:

                async def _upsert_node_txn(transaction):
                    # Start nested span for the transaction part
                    with tracer.start_as_current_span(
                        "upsert_node_transaction",
                        context=trace.set_span_in_context(span),
                    ) as txn_span:
                        transaction.insert_or_update(
                            table=self._node_table,
                            columns=(self._entity_id_col, self._properties_col),
                            values=[(node_id, properties_to_store)],
                        )
                        txn_span.set_attribute("db.spanner.rows", 1)

                await db.run_in_transaction_async(_upsert_node_txn)
                logger.debug(f"Upserted node '{node_id}'")
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error upserting node '{node_id}': {e}", exc_info=True)
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]
    ) -> None:
        """Creates/updates a directed edge using Spanner Mutation API."""
        db = self._get_db()
        tracer = self._get_tracer()

        async with tracer.start_as_current_span("upsert_edge") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "insert_or_update")
            span.set_attribute("db.spanner.table", self._edge_table)
            span.set_attribute("db.spanner.source_node_id", source_node_id)
            span.set_attribute("db.spanner.target_node_id", target_node_id)
            try:

                async def _upsert_edge_txn(transaction):
                    with tracer.start_as_current_span(
                        "upsert_edge_transaction",
                        context=trace.set_span_in_context(span),
                    ) as txn_span:
                        transaction.insert_or_update(
                            table=self._edge_table,
                            columns=(
                                self._edge_source_col,
                                self._edge_target_col,
                                self._properties_col,
                            ),
                            values=[(source_node_id, target_node_id, edge_data)],
                        )
                        txn_span.set_attribute("db.spanner.rows", 1)

                await db.run_in_transaction_async(_upsert_edge_txn)
                logger.debug(f"Upserted edge '{source_node_id}'->'{target_node_id}'")
                span.set_status(Status(StatusCode.OK))
            except google_exceptions.FailedPrecondition as e:
                span.set_status(Status(StatusCode.ERROR, f"Failed Precondition: {e}"))
                span.record_exception(e)
                if "ForeignKey" in str(e):
                    logger.error(
                        f"Error upserting edge '{source_node_id}'->'{target_node_id}': Source or Target node does not exist.",
                        exc_info=False,
                    )
                    raise ValueError(
                        f"Source node '{source_node_id}' or Target node '{target_node_id}' not found."
                    ) from e
                else:
                    logger.error(
                        f"Error upserting edge '{source_node_id}'->'{target_node_id}': {e}",
                        exc_info=True,
                    )
                    raise
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(
                    f"Error upserting edge '{source_node_id}'->'{target_node_id}': {e}",
                    exc_info=True,
                )
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def delete_node(self, node_id: str) -> None:
        """Deletes a node and its incident relationships within a transaction."""
        db = self._get_db()
        tracer = self._get_tracer()

        async with tracer.start_as_current_span("delete_node") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "delete_node_transaction")
            span.set_attribute("db.spanner.node_id", node_id)
            try:

                async def _delete_node_txn(transaction):
                    with tracer.start_as_current_span(
                        "delete_node_transaction_logic",
                        context=trace.set_span_in_context(span),
                    ) as txn_span:
                        # 1. Delete edges (Spanner handles FK cascade if set up, but explicit delete is safer)
                        edge_delete_sql = f"DELETE FROM {self._edge_table} WHERE {self._edge_source_col} = @node_id OR {self._edge_target_col} = @node_id"
                        params = {"node_id": node_id}
                        param_types = {
                            "node_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING)
                        }
                        rows_deleted_edges = await transaction.execute_update(
                            edge_delete_sql, params=params, param_types=param_types
                        )
                        txn_span.set_attribute(
                            "db.spanner.edges_deleted", rows_deleted_edges
                        )

                        # 2. Delete the node
                        node_delete_sql = f"DELETE FROM {self._node_table} WHERE {self._entity_id_col} = @node_id"
                        rows_deleted_node = await transaction.execute_update(
                            node_delete_sql, params=params, param_types=param_types
                        )
                        txn_span.set_attribute(
                            "db.spanner.node_deleted", rows_deleted_node
                        )

                await db.run_in_transaction_async(_delete_node_txn)
                logger.debug(f"Attempted deletion of node '{node_id}' and its edges.")
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error deleting node '{node_id}': {e}", exc_info=True)
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def remove_nodes(self, node_ids: list[str]) -> None:
        """Deletes multiple nodes and their incident relationships using Batch DML."""
        if not node_ids:
            return
        db = self._get_db()
        tracer = self._get_tracer()

        async with tracer.start_as_current_span("remove_nodes_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "batch_delete_nodes")
            span.set_attribute("db.spanner.node_count", len(node_ids))
            try:
                statements = []
                params = {"node_id_list": node_ids}
                param_types = {
                    "node_id_list": spanner_v1.Type(
                        array_element_type=spanner_v1.Type(
                            code=spanner_v1.TypeCode.STRING
                        )
                    )
                }

                edge_delete_sql = f"DELETE FROM {self._edge_table} WHERE {self._edge_source_col} IN UNNEST(@node_id_list) OR {self._edge_target_col} IN UNNEST(@node_id_list)"
                statements.append(
                    spanner_v1.BatchDmlStatement(
                        sql=edge_delete_sql, params=params, param_types=param_types
                    )
                )

                node_delete_sql = f"DELETE FROM {self._node_table} WHERE {self._entity_id_col} IN UNNEST(@node_id_list)"
                statements.append(
                    spanner_v1.BatchDmlStatement(
                        sql=node_delete_sql, params=params, param_types=param_types
                    )
                )

                async def _remove_nodes_txn(transaction):
                    with tracer.start_as_current_span(
                        "remove_nodes_batch_transaction",
                        context=trace.set_span_in_context(span),
                    ) as txn_span:
                        status, row_counts = await transaction.batch_update(
                            statements=statements
                        )
                        # Note: row_counts might not be precise for DELETE WHERE IN
                        txn_span.set_attribute("db.spanner.batch_status", str(status))

                await db.run_in_transaction_async(_remove_nodes_txn)
                logger.debug(f"Attempted batch deletion of nodes: {node_ids}")
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error removing nodes batch: {e}", exc_info=True)
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(_is_spanner_aborted),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """Deletes multiple directed edges using Batch DML."""
        if not edges:
            return
        db = self._get_db()
        tracer = self._get_tracer()

        async with tracer.start_as_current_span("remove_edges_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "batch_delete_edges")
            span.set_attribute("db.spanner.edge_count", len(edges))
            try:
                statements = []
                for src, tgt in edges:
                    sql = f"DELETE FROM {self._edge_table} WHERE {self._edge_source_col} = @source_id AND {self._edge_target_col} = @target_id"
                    params = {"source_id": src, "target_id": tgt}
                    param_types = {
                        "source_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
                        "target_id": spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
                    }
                    statements.append(
                        spanner_v1.BatchDmlStatement(
                            sql=sql, params=params, param_types=param_types
                        )
                    )

                async def _remove_edges_txn(transaction):
                    with tracer.start_as_current_span(
                        "remove_edges_batch_transaction",
                        context=trace.set_span_in_context(span),
                    ) as txn_span:
                        status, row_counts = await transaction.batch_update(
                            statements=statements
                        )
                        txn_span.set_attribute("db.spanner.batch_status", str(status))
                        # Sum row counts for total edges affected (approximate)
                        txn_span.set_attribute(
                            "db.spanner.edges_affected",
                            sum(row_counts) if row_counts else 0,
                        )

                await db.run_in_transaction_async(_remove_edges_txn)
                logger.debug(f"Attempted batch deletion of edges: {edges}")
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error removing edges batch: {e}", exc_info=True)
                raise

    # --- Batch Operations (Wrapped with Tracing) ---

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieves multiple nodes by their entity_ids using IN operator."""
        if not node_ids:
            return {}
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"SELECT {self._entity_id_col}, {self._properties_col} FROM {self._node_table} WHERE {self._entity_id_col} IN UNNEST(@node_id_list)"
        params = {"node_id_list": node_ids}
        param_types = {
            "node_id_list": spanner_v1.Type(
                array_element_type=spanner_v1.Type(code=spanner_v1.TypeCode.STRING)
            )
        }
        nodes_dict = {}

        async with tracer.start_as_current_span("get_nodes_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.requested_count", len(node_ids))
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        node_id = row[0]
                        props = row[1] if row[1] else {}
                        props[self._entity_id_col] = node_id
                        nodes_dict[node_id] = props
                span.set_attribute("db.spanner.found_count", len(nodes_dict))
                span.set_status(Status(StatusCode.OK))
                # Log missing nodes outside the span if needed
                for req_id in node_ids:
                    if req_id not in nodes_dict:
                        logger.debug(f"Node '{req_id}' not found in get_nodes_batch.")
                return nodes_dict
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting nodes batch: {e}", exc_info=True)
                raise

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieves degrees for multiple nodes in a batch."""
        if not node_ids:
            return {}
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"""
            WITH NodesToCount AS (SELECT entityId FROM UNNEST(@node_id_list) as entityId)
            SELECT n.entityId, (SELECT COUNT(*) FROM {self._edge_table} e WHERE e.{self._edge_source_col} = n.entityId OR e.{self._edge_target_col} = n.entityId) AS degree
            FROM NodesToCount n
        """
        params = {"node_id_list": node_ids}
        param_types = {
            "node_id_list": spanner_v1.Type(
                array_element_type=spanner_v1.Type(code=spanner_v1.TypeCode.STRING)
            )
        }
        degrees_dict = {node_id: 0 for node_id in node_ids}

        async with tracer.start_as_current_span("node_degrees_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.requested_count", len(node_ids))
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        degrees_dict[row[0]] = row[1]
                span.set_attribute(
                    "db.spanner.processed_count", len(degrees_dict)
                )  # Count nodes for which degree was calculated
                span.set_status(Status(StatusCode.OK))
                return degrees_dict
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting node degrees batch: {e}", exc_info=True)
                raise

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        tracer = self._get_tracer()
        async with tracer.start_as_current_span("edge_degrees_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.spanner.edge_pair_count", len(edge_pairs))
            if not edge_pairs:
                span.set_status(Status(StatusCode.OK))
                return {}

            try:
                # Collect all unique node IDs involved in the edge pairs
                node_ids_set: Set[str] = set()
                for src_id, tgt_id in edge_pairs:
                    node_ids_set.add(src_id)
                    node_ids_set.add(tgt_id)

                unique_node_ids = list(node_ids_set)

                # Get degrees for all unique nodes in a single batch call
                node_degrees_map = await self.node_degrees_batch(
                    unique_node_ids
                )  # This is already traced

                # Calculate edge degrees
                edge_degrees_result: Dict[Tuple[str, str], int] = {}
                for src_id, tgt_id in edge_pairs:
                    src_degree = node_degrees_map.get(src_id, 0)
                    tgt_degree = node_degrees_map.get(tgt_id, 0)
                    edge_degrees_result[(src_id, tgt_id)] = src_degree + tgt_degree

                span.set_attribute(
                    "db.spanner.calculated_edge_degree_count", len(edge_degrees_result)
                )
                span.set_status(Status(StatusCode.OK))
                return edge_degrees_result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting edge degrees batch: {e}", exc_info=True)
                raise

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieves properties for multiple edges using UNNEST."""
        if not pairs:
            return {}
        pool = self._get_pool()
        tracer = self._get_tracer()
        pair_structs = [{"source": p["src"], "target": p["tgt"]} for p in pairs]
        sql = f"""
            SELECT p.source, p.target, e.{self._properties_col}
            FROM UNNEST(@pairs_list) AS p
            JOIN {self._edge_table} AS e ON e.{self._edge_source_col} = p.source AND e.{self._edge_target_col} = p.target
        """
        params = {"pairs_list": pair_structs}
        param_types = {
            "pairs_list": spanner_v1.Type(
                code=spanner_v1.TypeCode.ARRAY,
                array_element_type=spanner_v1.Type(
                    code=spanner_v1.TypeCode.STRUCT,
                    struct_type=spanner_v1.StructType(
                        fields=[
                            spanner_v1.StructType.Field(
                                name="source",
                                type_=spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
                            ),
                            spanner_v1.StructType.Field(
                                name="target",
                                type_=spanner_v1.Type(code=spanner_v1.TypeCode.STRING),
                            ),
                        ]
                    ),
                ),
            )
        }
        final_edges_dict = {}

        async with tracer.start_as_current_span("get_edges_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.requested_count", len(pairs))
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        req_src, req_tgt, props = row[0], row[1], row[2]
                        original_pair = (req_src, req_tgt)
                        final_edges_dict[original_pair] = props if props else {}
                span.set_attribute("db.spanner.found_count", len(final_edges_dict))
                span.set_status(Status(StatusCode.OK))
                return final_edges_dict
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting edges batch: {e}", exc_info=True)
                raise

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Retrieves all connected edges for multiple nodes in a batch."""
        if not node_ids:
            return {}
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"""
            SELECT {self._edge_source_col}, {self._edge_target_col}
            FROM {self._edge_table}
            WHERE {self._edge_source_col} IN UNNEST(@node_id_list)
               OR {self._edge_target_col} IN UNNEST(@node_id_list)
        """
        params = {"node_id_list": node_ids}
        param_types = {
            "node_id_list": spanner_v1.Type(
                array_element_type=spanner_v1.Type(code=spanner_v1.TypeCode.STRING)
            )
        }
        nodes_edges_dict = {node_id: [] for node_id in node_ids}
        processed_undirected_edges = set()
        total_edges_found = 0

        async with tracer.start_as_current_span("get_nodes_edges_batch") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            span.set_attribute("db.spanner.requested_node_count", len(node_ids))
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(
                        sql, params=params, param_types=param_types
                    )
                    async for row in results:
                        src_node, connected_node = row[0], row[1]
                        if src_node and connected_node:
                            canonical_pair = tuple(sorted((src_node, connected_node)))
                            if canonical_pair not in processed_undirected_edges:
                                total_edges_found += 1
                                if src_node in nodes_edges_dict:
                                    nodes_edges_dict[src_node].append(
                                        (src_node, connected_node)
                                    )
                                if connected_node in nodes_edges_dict:
                                    nodes_edges_dict[connected_node].append(
                                        (connected_node, src_node)
                                    )
                                processed_undirected_edges.add(canonical_pair)
                span.set_attribute(
                    "db.spanner.total_edges_found", total_edges_found
                )  # Total unique edges involving requested nodes
                span.set_status(Status(StatusCode.OK))
                return nodes_edges_dict
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting nodes edges batch: {e}", exc_info=True)
                raise

    # --- Other Methods ---

    async def get_all_labels(self) -> list[str]:
        """Gets all distinct entity_ids from the Nodes table."""
        pool = self._get_pool()
        tracer = self._get_tracer()
        sql = f"SELECT DISTINCT {self._entity_id_col} FROM {self._node_table} ORDER BY {self._entity_id_col}"
        labels = []
        async with tracer.start_as_current_span("get_all_labels") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.statement", sql)
            try:
                async with pool.snapshot() as snapshot:
                    results = await snapshot.execute_sql(sql)
                    async for row in results:
                        labels.append(row[0])
                span.set_attribute("db.spanner.label_count", len(labels))
                span.set_status(Status(StatusCode.OK))
                return labels
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error getting all labels: {e}", exc_info=True)
                raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        """
        Retrieves a connected subgraph using iterative BFS-like queries.
        Handles '*' wildcard by fetching top nodes by degree.
        """
        pool = self._get_pool()
        tracer = self._get_tracer()
        kg = KnowledgeGraph()
        processed_nodes = set()  # Store entity_ids of nodes added to KG
        processed_edges = set()  # Store (src_id, tgt_id) tuples to avoid duplicates

        async with tracer.start_as_current_span("get_knowledge_graph") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.spanner.start_node", node_label)
            span.set_attribute("db.spanner.max_depth", max_depth)
            span.set_attribute("db.spanner.max_nodes", max_nodes)

            if node_label == "*":
                span.set_attribute("db.spanner.mode", "wildcard")
                # 1. Get top N nodes by degree
                degree_sql = f"""
                    WITH NodeDegrees AS (
                        SELECT N.{self._entity_id_col} AS node_id,
                               (SELECT COUNT(*) FROM {self._edge_table} E WHERE E.{self._edge_source_col} = N.{self._entity_id_col} OR E.{self._edge_target_col} = N.{self._entity_id_col}) AS degree
                        FROM {self._node_table} N
                    )
                    SELECT node_id
                    FROM NodeDegrees
                    ORDER BY degree DESC
                    LIMIT {max_nodes}
                """
                top_node_ids = []
                try:
                    async with pool.snapshot() as snapshot:
                        results = await snapshot.execute_sql(degree_sql)
                        async for row in results:
                            top_node_ids.append(row[0])
                except Exception as e:
                    logger.error(
                        f"Error getting top nodes by degree for '*': {e}", exc_info=True
                    )
                    span.set_status(
                        Status(StatusCode.ERROR, f"Failed getting top nodes: {e}")
                    )
                    span.record_exception(e)
                    return kg

                if not top_node_ids:
                    return kg
                span.set_attribute("db.spanner.top_node_count", len(top_node_ids))

                # 2. Get node details
                nodes_data = await self.get_nodes_batch(
                    top_node_ids
                )  # This method is already traced
                for node_id, props in nodes_data.items():
                    kg.nodes.append(
                        KnowledgeGraphNode(
                            id=node_id, labels=[node_id], properties=props
                        )
                    )
                    processed_nodes.add(node_id)

                # 3. Get edges between these top nodes
                if len(top_node_ids) > 1:
                    edges_sql = f"""
                        SELECT {self._edge_source_col}, {self._edge_target_col}, {self._properties_col}
                        FROM {self._edge_table}
                        WHERE {self._edge_source_col} IN UNNEST(@node_list)
                          AND {self._edge_target_col} IN UNNEST(@node_list)
                    """
                    params = {"node_list": top_node_ids}
                    param_types = {
                        "node_list": spanner_v1.Type(
                            array_element_type=spanner_v1.Type(
                                code=spanner_v1.TypeCode.STRING
                            )
                        )
                    }
                    try:
                        async with pool.snapshot() as snapshot:
                            results = await snapshot.execute_sql(
                                edges_sql, params=params, param_types=param_types
                            )
                            async for row in results:
                                src, tgt, props_json = row[0], row[1], row[2]
                                edge_tuple = tuple(sorted((src, tgt)))
                                if edge_tuple not in processed_edges:
                                    props = props_json if props_json else {}
                                    edge_id = f"{src}-{tgt}"
                                    kg.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type="DIRECTED",
                                            source=src,
                                            target=tgt,
                                            properties=props,
                                        )
                                    )
                                    processed_edges.add(edge_tuple)
                    except Exception as e:
                        logger.error(
                            f"Error getting edges for top nodes '*': {e}", exc_info=True
                        )
                        span.set_status(
                            Status(StatusCode.ERROR, f"Failed getting edges: {e}")
                        )
                        span.record_exception(e)

                kg.is_truncated = len(top_node_ids) >= max_nodes
                span.set_attribute("db.spanner.is_truncated", kg.is_truncated)

            else:  # Specific start node
                span.set_attribute("db.spanner.mode", "bfs_like")
                # Perform BFS-like expansion iteratively
                queue = {node_label}
                visited_nodes_for_bfs = set()
                # all_edges_found = set() # Removed unused variable

                for depth in range(max_depth + 1):
                    if not queue or len(processed_nodes) >= max_nodes:
                        break

                    current_level_nodes = list(queue)
                    visited_nodes_for_bfs.update(queue)
                    queue = set()

                    # Get node data for current level
                    nodes_to_fetch = [
                        nid for nid in current_level_nodes if nid not in processed_nodes
                    ]
                    if nodes_to_fetch:
                        fetched_nodes_data = await self.get_nodes_batch(
                            nodes_to_fetch
                        )  # Already traced
                        for node_id, props in fetched_nodes_data.items():
                            if len(processed_nodes) < max_nodes:
                                kg.nodes.append(
                                    KnowledgeGraphNode(
                                        id=node_id, labels=[node_id], properties=props
                                    )
                                )
                                processed_nodes.add(node_id)
                            else:
                                kg.is_truncated = True
                                break
                    if kg.is_truncated:
                        break

                    # Find neighbors for the next level (if not at max depth)
                    if depth < max_depth:
                        neighbors_sql = f"""
                            SELECT DISTINCT N2.{self._entity_id_col}
                            FROM {self._node_table} N1
                            JOIN {self._edge_table} E ON N1.{self._entity_id_col} = E.{self._edge_source_col} OR N1.{self._entity_id_col} = E.{self._edge_target_col}
                            JOIN {self._node_table} N2 ON (E.{self._edge_source_col} = N2.{self._entity_id_col} OR E.{self._edge_target_col} = N2.{self._entity_id_col}) AND N1.{self._entity_id_col} <> N2.{self._entity_id_col}
                            WHERE N1.{self._entity_id_col} IN UNNEST(@current_level_ids)
                        """
                        params = {"current_level_ids": current_level_nodes}
                        param_types = {
                            "current_level_ids": spanner_v1.Type(
                                array_element_type=spanner_v1.Type(
                                    code=spanner_v1.TypeCode.STRING
                                )
                            )
                        }
                        try:
                            async with pool.snapshot() as snapshot:
                                results = await snapshot.execute_sql(
                                    neighbors_sql,
                                    params=params,
                                    param_types=param_types,
                                )
                                async for row in results:
                                    neighbor_id = row[0]
                                    if neighbor_id not in visited_nodes_for_bfs:
                                        queue.add(neighbor_id)
                        except Exception as e:
                            logger.error(
                                f"Error finding neighbors at depth {depth + 1}: {e}",
                                exc_info=True,
                            )
                            span.set_status(
                                Status(
                                    StatusCode.ERROR, f"Failed finding neighbors: {e}"
                                )
                            )
                            span.record_exception(e)
                            break  # Stop BFS on error

                # After BFS, fetch all edges involving the processed nodes
                if processed_nodes:
                    edges_sql = f"""
                        SELECT {self._edge_source_col}, {self._edge_target_col}, {self._properties_col}
                        FROM {self._edge_table}
                        WHERE {self._edge_source_col} IN UNNEST(@node_list)
                          AND {self._edge_target_col} IN UNNEST(@node_list)
                    """
                    params = {"node_list": list(processed_nodes)}
                    param_types = {
                        "node_list": spanner_v1.Type(
                            array_element_type=spanner_v1.Type(
                                code=spanner_v1.TypeCode.STRING
                            )
                        )
                    }
                    try:
                        async with pool.snapshot() as snapshot:
                            results = await snapshot.execute_sql(
                                edges_sql, params=params, param_types=param_types
                            )
                            async for row in results:
                                src, tgt, props_json = row[0], row[1], row[2]
                                edge_tuple = tuple(sorted((src, tgt)))
                                if edge_tuple not in processed_edges:
                                    props = props_json if props_json else {}
                                    edge_id = f"{src}-{tgt}"
                                    kg.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type="DIRECTED",
                                            source=src,
                                            target=tgt,
                                            properties=props,
                                        )
                                    )
                                    processed_edges.add(edge_tuple)
                    except Exception as e:
                        logger.error(
                            f"Error getting final edges for '{node_label}': {e}",
                            exc_info=True,
                        )
                        span.set_status(
                            Status(StatusCode.ERROR, f"Failed getting final edges: {e}")
                        )
                        span.record_exception(e)

                # Set truncation flag if queue still had nodes when max_nodes was hit
                if queue and len(processed_nodes) >= max_nodes:
                    kg.is_truncated = True
                span.set_attribute("db.spanner.is_truncated", kg.is_truncated)

            # Set final status for the overall operation span
            if span.status.status_code != StatusCode.ERROR:
                span.set_status(Status(StatusCode.OK))

            logger.info(
                f"Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)} | Truncated: {kg.is_truncated}"
            )
            return kg

    async def index_done_callback(self) -> None:
        """Spanner commits transactions automatically, so this is likely a no-op."""
        logger.debug(
            "index_done_callback called for SpannerGraphStorage. No explicit action taken (relying on Spanner transactions)."
        )
        pass

    async def drop(self) -> dict[str, str]:
        """Drops all data from the graph tables using DELETE statements."""
        db = self._get_db()
        tracer = self._get_tracer()
        logger.warning(
            f"Dropping all data from Spanner graph tables: {self._edge_table}, {self._node_table}"
        )

        async with tracer.start_as_current_span("drop_spanner_graph_data") as span:
            span.set_attribute("db.system", "spanner")
            span.set_attribute("db.operation", "batch_delete_all")
            span.set_attribute(
                "db.spanner.tables", [self._edge_table, self._node_table]
            )
            try:
                # Must delete edges before nodes due to potential FK constraints (even with ON DELETE CASCADE, explicit is safer)
                statements = [
                    spanner_v1.BatchDmlStatement(
                        sql=f"DELETE FROM {self._edge_table} WHERE true"
                    ),
                    spanner_v1.BatchDmlStatement(
                        sql=f"DELETE FROM {self._node_table} WHERE true"
                    ),
                ]

                async def _drop_txn(transaction):
                    with tracer.start_as_current_span(
                        "drop_data_transaction", context=trace.set_span_in_context(span)
                    ) as txn_span:
                        status, row_counts = await transaction.batch_update(
                            statements=statements
                        )
                        txn_span.set_attribute("db.spanner.batch_status", str(status))
                        txn_span.set_attribute(
                            "db.spanner.rows_affected_edges",
                            row_counts[0] if row_counts else -1,
                        )
                        txn_span.set_attribute(
                            "db.spanner.rows_affected_nodes",
                            row_counts[1] if len(row_counts) > 1 else -1,
                        )

                await db.run_in_transaction_async(_drop_txn)
                logger.info("Successfully dropped data from Spanner graph tables.")
                span.set_status(Status(StatusCode.OK))
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Error dropping Spanner graph data: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
