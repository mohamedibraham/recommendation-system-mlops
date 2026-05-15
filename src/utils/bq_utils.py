import logging
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, WriteDisposition, QueryJobConfig

logger = logging.getLogger(__name__)


class BigQueryManager:
    """
    Production-grade BigQuery client wrapper.

    Provides:
    - Query execution → DataFrame
    - DataFrame → BigQuery table (write / append / truncate)
    - DDL execution (CREATE TABLE, etc.)
    - Job-level configuration (timeout, priority, labels)
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        logger.info("BigQueryManager initialized for project: %s", project_id)

    
    # READ
    
    def query_to_dataframe(
        self,
        query: str,
        timeout: int = 300,
        job_config: Optional[QueryJobConfig] = None,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a pandas DataFrame.

        Args:
            query:      Standard SQL string.
            timeout:    Maximum seconds to wait for the job to complete.
            job_config: Optional BigQuery QueryJobConfig for advanced settings.

        Returns:
            pd.DataFrame with query results.

        Raises:
            google.cloud.exceptions.GoogleCloudError on BQ-side failures.
        """
        logger.debug("Executing query:\n%s", query[:300])
        job = self.client.query(query, job_config=job_config, timeout=timeout)
        df: pd.DataFrame = job.to_dataframe(timeout=timeout)   # ← parentheses required
        logger.info("Query returned %d rows, %d columns.", len(df), len(df.columns))
        return df

    # Keep old snake_case alias for backward compatibility with data_validation.py
    Query_To_Dataframe = query_to_dataframe


    # WRITE

    def dataframe_to_table(
        self,
        df: pd.DataFrame,
        destination_table: str,          # "project.dataset.table"  or "dataset.table"
        write_disposition: str = "WRITE_TRUNCATE",
        schema: Optional[list] = None,
        timeout: int = 600,
    ) -> None:
        """
        Upload a pandas DataFrame to a BigQuery table.

        Args:
            df:                  DataFrame to upload.
            destination_table:   Full table ref, e.g. "myproject.myds.mytable".
            write_disposition:   WRITE_TRUNCATE | WRITE_APPEND | WRITE_EMPTY.
            schema:              Optional list of bigquery.SchemaField objects.
            timeout:             Job timeout in seconds.
        """
        cfg = LoadJobConfig(
            write_disposition=WriteDisposition[write_disposition],
            schema=schema or [],
            autodetect=(schema is None),
        )
        job = self.client.load_table_from_dataframe(
            df, destination_table, job_config=cfg
        )
        job.result(timeout=timeout)
        logger.info(
            "Loaded %d rows → %s (%s).", len(df), destination_table, write_disposition
        )

    
    # DDL helpers

    def execute_ddl(self, ddl: str, timeout: int = 120) -> None:
        """Run a DDL statement (CREATE / ALTER / DROP …)."""
        self.client.query(ddl, timeout=timeout).result(timeout=timeout)
        logger.info("DDL executed successfully.")

    def table_exists(self, table_ref: str) -> bool:
        """Return True if the table exists in BigQuery."""
        try:
            self.client.get_table(table_ref)
            return True
        except Exception:
            return False