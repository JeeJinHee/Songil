import enum

from pydantic import Field
from app.models.session import CurrentStep
from app.schemas.common import CamelModel, UtcDatetime
from app.models.guide import SelectionResult

class NextTargetResult(str, enum.Enum):
    TARGET = "TARGET"
    NEED_CHOICE = "NEED_CHOICE"
    SOLD_OUT = "SOLD_OUT"
    NOT_FOUND = "NOT_FOUND"
    DONE = "DONE"

class Button(CamelModel):
    button_id: str
    text: str
    type: str | None = None         # "tab"/"menu"/"etc"/없음
    group: str | None = None
    sold_out: bool = False
    bbox: list[float] | None = None         # [x1, y1, x2, y2]

class CartRow(CamelModel):
    texts: list[str]

class NextTargetRequest(CamelModel):
    screen_seq: int = Field(ge=0)
    captured_at: UtcDatetime
    title: str | None = None
    buttons: list[Button]
    cart_rows: list[CartRow] = []
    answer: str | None = None

class Target(CamelModel):
    button_id: str
    text: str

class NextTargetResponse(CamelModel):
    screen_seq: int
    result: NextTargetResult
    target: Target | None = None
    choices: list[str] | None = None
    current_step: CurrentStep
    message: str
    source: str | None=None
    reason: str | None=None

class SelectionResultRequest(CamelModel):
    target_button_id: str
    screen_seq: int = Field(ge=0)
    screen_changed: bool
    title: str | None=None
    buttons: list[Button] = []
    cart_rows: list[CartRow] = []

class SelectionResultResponse(CamelModel):
    result: SelectionResult
    current_step: CurrentStep
    message: str
