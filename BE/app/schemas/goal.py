import enum

from pydantic import Field
from app.models.session import CurrentStep
from app.schemas.common import CamelModel

class GoalResultStatus(str, enum.Enum):
    NEED_CONFIRM = "NEED_CONFIRM"
    NOT_UNDERSTOOD = "NOT_UNDERSTOOD"
    CONFIRMED = "CONFIRMED"
    CLEARED = "CLEARED"
    BROWSING = "BROWSING"

class GoalConfirmAction(str, enum.Enum):
    CONFIRM = "CONFIRM"
    RETRY = "RETRY"
    BROWSE = "BROWSE"

class Goal(CamelModel):
    menu: str | None
    options: dict[str, str]
    quantity: int = Field(ge=1)

class GoalCreateRequest(CamelModel):
    text: str = Field(min_length=1)

class GoalCreateResponse(CamelModel):
    goal: Goal | None
    goal_status: GoalResultStatus
    message: str

class GoalConfirmRequest(CamelModel):
    action: GoalConfirmAction

class GoalConfirmResponse(CamelModel):
    goal_status: GoalResultStatus
    current_step: CurrentStep
    message: str

class GoalResponse(CamelModel):
    goal: Goal
    goal_status: GoalResultStatus
