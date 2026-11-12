import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, SQLModel, Session, create_engine

from src.plugin_runtime.capabilities.data import RuntimeDataCapabilityMixin
from src.services import database_service


class PaginationRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    value: str


@pytest.mark.asyncio
async def test_database_get_applies_offset(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[PaginationRecord.__table__])
    with Session(engine) as session:
        session.add_all([PaginationRecord(value=str(i)) for i in range(5)])
        session.commit()
    monkeypatch.setattr(database_service, "get_db_session", lambda **kwargs: Session(engine))
    rows = await database_service.db_get(PaginationRecord, order_by="id", limit=2, offset=2)
    assert [row["value"] for row in rows] == ["2", "3"]


def test_offset_validation():
    assert RuntimeDataCapabilityMixin._database_offset({"offset": 3, "order_by": ["id"]}) == 3
    for args in ({"offset": -1}, {"offset": True}, {"offset": 1}, {"offset": "1"}):
        with pytest.raises(ValueError):
            RuntimeDataCapabilityMixin._database_offset(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["_cap_database_get", "_cap_database_query"])
async def test_capabilities_forward_offset(monkeypatch, method):
    from src.common.database import database_model

    monkeypatch.setattr(database_model, "PaginationRecord", PaginationRecord, raising=False)
    seen = {}

    async def get(**kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(database_service, "db_get", get)
    capability = RuntimeDataCapabilityMixin()
    result = await getattr(capability, method)("test.plugin", "database.get", {
        "model_name": "PaginationRecord", "limit": 2, "order_by": ["id"], "offset": 3,
    })
    assert result == []
    assert seen["offset"] == 3
