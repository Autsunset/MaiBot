from unittest.mock import Mock

import importlib

import pytest

from src.common.data_models.message_component_data_model import ImageComponent
from src.maisaka.visual import chat_history_refresher as visual


@pytest.mark.parametrize("pending", [False, True])
def test_pending_state_comes_from_live_jobs_not_unfinished_records(monkeypatch, pending):
    module = importlib.import_module("src.chat.image_system.image_manager")
    manager = Mock()
    manager.is_description_pending.return_value = pending
    monkeypatch.setattr(module, "image_manager", manager)
    session = Mock(side_effect=AssertionError("Pending state should not query SQLite"))
    monkeypatch.setattr(visual, "get_db_session", session)
    component = ImageComponent(binary_hash="test-image", binary_data=b"synthetic bytes")

    assert visual._is_image_component_pending(component) is pending
    manager.is_description_pending.assert_called_once_with(component.binary_hash)
    session.assert_not_called()


def test_failed_image_gets_neutral_placeholder_without_repeated_refresh(monkeypatch):
    component = ImageComponent(binary_hash="test-image")
    monkeypatch.setattr(visual, "_lookup_cached_image_description", lambda image_hash: "")
    monkeypatch.setattr(visual, "_is_image_description_pending", lambda image_hash: False)

    assert visual._refresh_pending_visual_components([component])
    assert component.content == "[图片：暂无可用的画面内容]"
    assert not visual._refresh_pending_visual_components([component])


def test_active_image_keeps_waiting_and_completed_description_replaces_placeholder(monkeypatch):
    component = ImageComponent(binary_hash="test-image")
    monkeypatch.setattr(visual, "_lookup_cached_image_description", lambda image_hash: "")
    monkeypatch.setattr(visual, "_is_image_description_pending", lambda image_hash: True)
    assert not visual._refresh_pending_visual_components([component])
    assert component.content == ""

    component.content = "[图片：暂无可用的画面内容]"
    monkeypatch.setattr(visual, "_lookup_cached_image_description", lambda image_hash: "a visible object")
    assert visual._refresh_pending_visual_components([component])
    assert component.content == "[图片：a visible object]"


def test_failed_job_notifies_monitor_and_clears_tracking(monkeypatch):
    callback = Mock()
    monkeypatch.setattr(visual, "_PLANNER_PENDING_IMAGE_HASHES", {"test-image"})
    monkeypatch.setattr(visual, "_MONITOR_PENDING_IMAGE_REFRESHERS", {"test-image": [callback]})
    logs = []
    monkeypatch.setattr(visual.logger, "info", logs.append)

    visual.log_tracked_image_recognition_completed("test-image", succeeded=False)

    assert not visual._PLANNER_PENDING_IMAGE_HASHES
    assert not visual._MONITOR_PENDING_IMAGE_REFRESHERS
    callback.assert_called_once_with("test-image")
    assert "succeeded=False" in logs[0]
    assert "已完成识别" not in logs[0]
