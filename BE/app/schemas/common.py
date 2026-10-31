from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """JSON 필드는 camelCase, 파이썬 코드에서는 snake_case로 사용한다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)


def _ensure_utc(value: datetime) -> datetime:
    # SQLite는 시간대 정보를 저장하지 않으므로, 시간대가 없으면 UTC로 간주한다
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


UtcDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
