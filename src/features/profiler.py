import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Automated feature profiler that yields a dictionary of critical 
    supply quality metrics and data distribution metadata.
    """
    if df.empty:
        return {"status": "error", "message": "DataFrame is empty"}
        
    report = {
        "status": "success",
        "row_count": len(df),
        "global_ctr": df['actual_click'].mean() if 'actual_click' in df.columns else None,
        "missing_values": df.isnull().sum().to_dict(),
        "device_breakdown": df['device_type'].value_counts(normalize=True).to_dict() if 'device_type' in df.columns else {},
        "avg_cost": df['cost'].mean() if 'cost' in df.columns else None,
        "avg_revenue": df['revenue'].mean() if 'revenue' in df.columns else None
    }
    
    return report
