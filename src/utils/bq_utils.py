import logging
from typing import Optional
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery import LoadJobConfig, WriteDisposition, QueryJobConfig


logger = logging.getLogger(__name__)


class BigQueryManager:

    def __init__(self, project_id: str) -> None:

        self.project_id = project_id

        self.client = bigquery.Client(project=project_id)

        logger.info("BigQueryManager initialized for project: %s", project_id)


    def Query_To_Dataframe(
        self,
        query: str,
        timeout: int = 300,
        job_config: Optional[QueryJobConfig] = None,
    ) -> pd.DataFrame:

        logger.debug("Executing query:\n%s", query[:300])
        job = self.client.query(query, job_config=job_config)
        job.result(timeout=timeout)  # ← انتظر اكتمال الـ execution
        df = job.to_dataframe(
           create_bqstorage_client=True,  # أسرع بكثير للجداول الكبيرة
           timeout=timeout
         )
        logger.info("Query returned %d rows, %d columns.", len(df), len(df.columns))
        return df

    def dataframe_to_table(
        self,
        df: pd.DataFrame,
        destination_table: str,          
        write_disposition: str = "WRITE_TRUNCATE",
        schema: Optional[list] = None,
        timeout: int = 600,
    ) -> None:

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

    def execute_ddl(self, ddl: str, timeout: int = 120) -> None:
        self.client.query(ddl, timeout=timeout).result(timeout=timeout)
        logger.info("DDL executed successfully.")

    def table_exists(self, table_ref: str) -> bool:
        try:
            self.client.get_table(table_ref)
            return True
        except Exception:

            return False 

