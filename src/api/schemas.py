"""
Pydantic Schemas for FastAPI Inference Service.
"""

from pydantic import BaseModel, Field

class BidRequest(BaseModel):
    """Payload representing an incoming ad auction request."""
    request_id: str = Field(..., description="Unique auction identifier")
    campaign_id: str = Field(..., description="Target campaign ID")
    ad_id: str = Field(..., description="Target ad ID")
    user_id: str = Field(..., description="Unique user identifier")
    publisher_id: str = Field(..., description="Domain/app where ad will show")
    exchange_id: str = Field(..., description="Ad exchange broadcasting the request")
    device_type: str = Field(..., description="User device (mobile, desktop, tablet)")
    os: str = Field(..., alias="os_family", description="User operating system")
    country: str = Field(..., description="User location")
    time_of_day: str = Field(..., description="Hour of the day as a string (e.g. '08:00')")
    
    hour_of_day: int = Field(0, description="Hour of the day (0-23)")
    day_of_week: int = Field(0, description="Day of the week (0=Mon, 6=Sun)")
    floor_price: float = Field(..., description="Minimum bid price to participate")

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_1001",
                "campaign_id": "camp_1",
                "ad_id": "ad_12",
                "user_id": "user_4567",
                "publisher_id": "pub_11",
                "exchange_id": "ex_appnexus",
                "device_type": "mobile",
                "os_family": "android",
                "country": "US",
                "time_of_day": "14:30",
                "hour_of_day": 14,
                "floor_price": 0.85
            }
        }


class BidResponse(BaseModel):
    """Response payload returned back to the ad exchange bidding engine."""
    request_id: str = Field(..., description="Original requested auction ID")
    predicted_ctr: float = Field(..., description="AI estimated click-through rate probability")
    bid_price: float = Field(..., description="Calculated actual bid value in USD")
    bid_placed: bool = Field(..., description="Whether the algorithmic bid cleared the exchange floor price")
    strategy_used: str = Field(..., description="Logic unit deployed (e.g., ValueBasedBidder)")
    processing_time_ms: float = Field(..., description="Latency of inference execution")
