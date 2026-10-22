"""Pydantic models that validate API input and describe API output."""
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """Customer fields. Only tenure, monthly_charges and contract are required;
    other missing fields are filled with the most common training value."""
    tenure: int = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, le=1000)
    contract: Literal["Month-to-month", "One year", "Two year"]
    total_charges: Optional[float] = Field(None, ge=0, le=200000)
    gender: Optional[Literal["Male", "Female"]] = None
    senior_citizen: Optional[YesNo] = None
    partner: Optional[YesNo] = None
    dependents: Optional[YesNo] = None
    phone_service: Optional[YesNo] = None
    multiple_lines: Optional[Literal["Yes", "No", "No phone service"]] = None
    internet_service: Optional[Literal["DSL", "Fiber optic", "No"]] = None
    online_security: Optional[YesNoNoInternet] = None
    online_backup: Optional[YesNoNoInternet] = None
    device_protection: Optional[YesNoNoInternet] = None
    tech_support: Optional[YesNoNoInternet] = None
    streaming_tv: Optional[YesNoNoInternet] = None
    streaming_movies: Optional[YesNoNoInternet] = None
    paperless_billing: Optional[YesNo] = None
    payment_method: Optional[Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]] = None

    @field_validator("senior_citizen", mode="before")
    @classmethod
    def senior_as_yes_no(cls, v):
        # allow 0/1/true/false as well as Yes/No
        if v in (0, False):
            return "No"
        if v in (1, True):
            return "Yes"
        return v

    def to_model_row(self) -> dict:
        """Convert to the Telco column names the pipeline expects."""
        names = {
            "tenure": "tenure", "monthly_charges": "MonthlyCharges", "contract": "Contract",
            "total_charges": "TotalCharges", "gender": "gender", "senior_citizen": "SeniorCitizen",
            "partner": "Partner", "dependents": "Dependents", "phone_service": "PhoneService",
            "multiple_lines": "MultipleLines", "internet_service": "InternetService",
            "online_security": "OnlineSecurity", "online_backup": "OnlineBackup",
            "device_protection": "DeviceProtection", "tech_support": "TechSupport",
            "streaming_tv": "StreamingTV", "streaming_movies": "StreamingMovies",
            "paperless_billing": "PaperlessBilling", "payment_method": "PaymentMethod",
        }
        data = self.model_dump()
        return {names[k]: v for k, v in data.items()}


class Factor(BaseModel):
    feature: str
    value: str
    effect_on_probability: float
    direction: str


class Suggestion(BaseModel):
    action: str
    reason: str


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    risk_level: str
    important_factors: list[Factor]
    protective_factors: list[Factor]
    retention_suggestions: list[Suggestion]
    note: str = ("Important factors describe what influences the model prediction; "
                 "they do not prove what causes churn.")
    disclaimer: str


class BatchResponse(BaseModel):
    n_customers: int
    warnings: list[str]
    summary: dict
    results: list[dict]
