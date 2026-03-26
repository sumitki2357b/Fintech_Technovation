from __future__ import annotations

from typing import Optional
import numpy as np

from app.core.pipeline import FraudFeatureEngineer
from app.schemas.response import PipelineResult


def run_feature_pipeline(csv_path: str, output_path: Optional[str] = None) -> PipelineResult:
    dataframe, summary = FraudFeatureEngineer(csv_path=csv_path).run()

    # 🔧 FIX: make data JSON-safe (this solves your 500 error)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.fillna(0)

    if output_path:
        dataframe.to_csv(output_path, index=False)

    return {
        "rows_processed": int(len(dataframe)),
        "columns_produced": list(dataframe.columns),
        "summary": summary,
        "data": dataframe.head(50).to_dict(orient="records"),
    }
