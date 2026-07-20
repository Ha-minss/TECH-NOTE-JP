import os

from recover24.providers.deepseek import DeepSeekProvider


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true}'
                    }
                }
            ]
        }


def test_deepseek_provider_extracts_message_content(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import recover24.providers.deepseek as module

    monkeypatch.setattr(module.requests, "post", fake_post)

    provider = DeepSeekProvider(timeout_seconds=7)
    result = provider.generate_json("Return JSON")

    assert result == '{"ok": true}'
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "deepseek-chat"
    assert captured["timeout"] == 7
