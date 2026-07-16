# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import json
import os

import pytest

from block_detector_protocol import (
    DetectorClient,
    ProtocolError,
    read_message,
    write_message,
)


def _response(**overrides):
    response = {
        "id": 1,
        "ok": True,
        "target": "fire",
        "class_id": 0,
        "class_name": "灭火装置",
        "confidence": 0.95,
        "box": [10.0, 20.0, 110.0, 120.0],
    }
    response.update(overrides)
    return response


def _client_with_responses(*responses):
    response_stream = io.StringIO(
        "".join(json.dumps(response, ensure_ascii=True) + "\n" for response in responses)
    )
    request_stream = io.StringIO()
    return DetectorClient(request_stream, response_stream), request_stream


class _TrackingReader(io.StringIO):
    def __init__(self, initial_value):
        io.StringIO.__init__(self, initial_value)
        self.readline_calls = 0

    def readline(self, *args, **kwargs):
        self.readline_calls += 1
        return io.StringIO.readline(self, *args, **kwargs)


class _FailFirstFlushStream(object):
    def __init__(self):
        self.writes = []
        self.flush_calls = 0

    def write(self, data):
        self.writes.append(data)

    def flush(self):
        self.flush_calls += 1
        if self.flush_calls == 1:
            raise IOError("simulated flush failure")


def test_unicode_message_round_trip_on_binary_stream():
    stream = io.BytesIO()

    write_message(stream, {"name": "灭火装置"})
    stream.seek(0)

    assert read_message(stream) == {"name": "灭火装置"}
    assert stream.getvalue().endswith(b"\n")
    assert all(byte < 128 for byte in bytearray(stream.getvalue()))


@pytest.mark.parametrize("binary", [False, True], ids=["text", "binary"])
def test_real_pipe_streams_are_flushed_and_readable(binary):
    read_fd, write_fd = os.pipe()
    read_mode, write_mode = ("rb", "wb") if binary else ("r", "w")
    reader = os.fdopen(read_fd, read_mode)
    writer = os.fdopen(write_fd, write_mode)
    try:
        write_message(writer, {"ready": True})
        assert read_message(reader) == {"ready": True}
    finally:
        writer.close()
        reader.close()


def test_read_message_raises_eof_error_at_end_of_stream():
    with pytest.raises(EOFError):
        read_message(io.StringIO(""))


@pytest.mark.parametrize("line", ["   \n", "not-json\n", "[1,2]\n"])
def test_read_message_rejects_blank_malformed_and_non_object(line):
    with pytest.raises(ProtocolError):
        read_message(io.StringIO(line))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_write_message_rejects_non_finite_json_numbers(value):
    stream = io.StringIO()

    with pytest.raises(ProtocolError):
        write_message(stream, {"value": value})

    assert stream.getvalue() == ""


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_read_message_rejects_non_finite_json_constants(constant):
    with pytest.raises(ProtocolError):
        read_message(io.StringIO('{"value":%s}\n' % constant))


def test_write_message_rejects_non_object_and_unserializable_payload():
    with pytest.raises(ProtocolError):
        write_message(io.StringIO(), ["not", "an", "object"])
    with pytest.raises(ProtocolError):
        write_message(io.StringIO(), {"raw": object()})


def test_detect_writes_request_and_returns_valid_response():
    client, requests = _client_with_responses(_response())

    result = client.detect("/tmp/现场.jpg", "fire")

    assert result["class_name"] == "灭火装置"
    requests.seek(0)
    assert read_message(requests) == {
        "id": 1,
        "image_path": "/tmp/现场.jpg",
        "target": "fire",
    }


@pytest.mark.parametrize("response_id", [2, True])
def test_detect_rejects_mismatched_or_boolean_response_id(response_id):
    client, unused = _client_with_responses(_response(id=response_id))

    with pytest.raises(ProtocolError, match="id"):
        client.detect("/tmp/image.jpg", "fire")


@pytest.mark.parametrize("overrides", [{"ok": "yes"}, {"ok": None}])
def test_detect_requires_strict_boolean_ok(overrides):
    response = _response()
    if overrides["ok"] is None:
        del response["ok"]
    else:
        response.update(overrides)
    client, unused = _client_with_responses(response)

    with pytest.raises(ProtocolError, match="ok"):
        client.detect("/tmp/image.jpg", "fire")


def test_detect_propagates_non_empty_server_error():
    client, unused = _client_with_responses(
        {"id": 1, "ok": False, "error": "model unavailable"}
    )

    with pytest.raises(ProtocolError, match="model unavailable"):
        client.detect("/tmp/image.jpg", "fire")


def test_detect_can_continue_after_matched_business_error():
    client, requests = _client_with_responses(
        {"id": 1, "ok": False, "error": "model unavailable"},
        _response(id=2),
    )

    with pytest.raises(ProtocolError, match="model unavailable"):
        client.detect("/tmp/one.jpg", "fire")
    assert client.detect("/tmp/two.jpg", "fire")["id"] == 2

    requests.seek(0)
    assert read_message(requests)["id"] == 1
    assert read_message(requests)["id"] == 2


def test_flush_failure_reserves_id_and_poisons_client_without_more_io():
    requests = _FailFirstFlushStream()
    responses = _TrackingReader(json.dumps(_response(id=2)) + "\n")
    client = DetectorClient(requests, responses)

    with pytest.raises(ProtocolError, match="flush"):
        client.detect("/tmp/one.jpg", "fire")

    assert client._next_request_id == 2
    assert len(requests.writes) == 1
    assert responses.readline_calls == 0

    with pytest.raises(ProtocolError, match="unusable|poisoned"):
        client.detect("/tmp/two.jpg", "fire")

    assert client._next_request_id == 2
    assert len(requests.writes) == 1
    assert responses.readline_calls == 0


@pytest.mark.parametrize("response_text", ["", "not-json\n"], ids=["eof", "invalid-json"])
def test_read_eof_or_invalid_json_poisons_client(response_text):
    requests = io.StringIO()
    responses = _TrackingReader(response_text)
    client = DetectorClient(requests, responses)

    with pytest.raises((EOFError, ProtocolError)):
        client.detect("/tmp/one.jpg", "fire")

    first_request = requests.getvalue()
    assert client._next_request_id == 2
    assert responses.readline_calls == 1

    with pytest.raises(ProtocolError, match="unusable|poisoned"):
        client.detect("/tmp/two.jpg", "fire")

    assert client._next_request_id == 2
    assert requests.getvalue() == first_request
    assert responses.readline_calls == 1


def test_mismatched_response_id_poisons_client():
    client, requests = _client_with_responses(_response(id=99), _response(id=2))

    with pytest.raises(ProtocolError, match="id"):
        client.detect("/tmp/one.jpg", "fire")
    first_request = requests.getvalue()

    with pytest.raises(ProtocolError, match="unusable|poisoned"):
        client.detect("/tmp/two.jpg", "fire")

    assert client._next_request_id == 2
    assert requests.getvalue() == first_request


@pytest.mark.parametrize("error", [None, "", "   "])
def test_detect_rejects_failed_response_without_non_empty_error(error):
    response = {"id": 1, "ok": False}
    if error is not None:
        response["error"] = error
    client, unused = _client_with_responses(response)

    with pytest.raises(ProtocolError, match="error"):
        client.detect("/tmp/image.jpg", "fire")


@pytest.mark.parametrize("class_id", [-1, True, 1.5, "1"])
def test_detect_rejects_invalid_class_id(class_id):
    client, unused = _client_with_responses(_response(class_id=class_id))

    with pytest.raises(ProtocolError, match="class_id"):
        client.detect("/tmp/image.jpg", "fire")


@pytest.mark.parametrize("class_name", [None, "", "  "])
def test_detect_rejects_missing_or_empty_class_name(class_name):
    response = _response()
    if class_name is None:
        del response["class_name"]
    else:
        response["class_name"] = class_name
    client, unused = _client_with_responses(response)

    with pytest.raises(ProtocolError, match="class_name"):
        client.detect("/tmp/image.jpg", "fire")


@pytest.mark.parametrize(
    "confidence",
    [float("nan"), float("inf"), 10 ** 1000, -0.01, 1.01, True, "0.9"],
)
def test_detect_rejects_non_finite_or_out_of_range_confidence(confidence):
    client, unused = _client_with_responses(_response(confidence=confidence))

    with pytest.raises(ProtocolError):
        client.detect("/tmp/image.jpg", "fire")


@pytest.mark.parametrize(
    "box",
    [
        [10, 20, float("nan"), 120],
        [10, 20, float("inf"), 120],
        [10, 20, 10 ** 1000, 120],
        [10, 20, 110],
        [10, 20, 10, 120],
        [10, 20, 110, 20],
        [10, 20, True, 120],
        [10, 20, "110", 120],
    ],
)
def test_detect_rejects_invalid_or_degenerate_box(box):
    client, unused = _client_with_responses(_response(box=box))

    with pytest.raises(ProtocolError):
        client.detect("/tmp/image.jpg", "fire")


def test_detect_rejects_target_mismatch():
    client, unused = _client_with_responses(_response(target="gas"))

    with pytest.raises(ProtocolError, match="target"):
        client.detect("/tmp/image.jpg", "fire")


def test_detect_request_ids_increment_across_calls():
    client, requests = _client_with_responses(_response(), _response(id=2))

    client.detect("/tmp/one.jpg", "fire")
    client.detect("/tmp/two.jpg", "fire")

    requests.seek(0)
    assert read_message(requests)["id"] == 1
    assert read_message(requests)["id"] == 2


@pytest.mark.parametrize(
    "image_path,target",
    [
        ("", "fire"),
        ("   ", "fire"),
        (None, "fire"),
        ("/tmp/image.jpg", ""),
        ("/tmp/image.jpg", "   "),
        ("/tmp/image.jpg", None),
    ],
)
def test_detect_rejects_empty_image_or_target_without_writing(image_path, target):
    client, requests = _client_with_responses(_response())

    with pytest.raises(ProtocolError):
        client.detect(image_path, target)

    assert requests.getvalue() == ""
