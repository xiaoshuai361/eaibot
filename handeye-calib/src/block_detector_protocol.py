from __future__ import absolute_import, division, print_function

import json
import math


try:
    text_type = unicode
except NameError:
    text_type = str

try:
    string_types = (basestring,)
except NameError:
    string_types = (str,)

try:
    integer_types = (int, long)
except NameError:
    integer_types = (int,)

try:
    binary_type = bytes
except NameError:
    binary_type = str


class ProtocolError(RuntimeError):
    pass


def _write_ascii_line(stream, line):
    """Write an ASCII JSON line to either a text or binary stream."""
    try:
        stream.write(line)
    except TypeError:
        try:
            stream.write(line.encode("ascii"))
        except Exception as exc:
            raise ProtocolError("Could not write protocol message: %s" % exc)
    except Exception as exc:
        raise ProtocolError("Could not write protocol message: %s" % exc)

    try:
        stream.flush()
    except Exception as exc:
        raise ProtocolError("Could not flush protocol message: %s" % exc)


def write_message(stream, payload):
    if not isinstance(payload, dict):
        raise ProtocolError("Protocol payload must be a JSON object")

    try:
        message = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Protocol payload is not JSON serializable: %s" % exc)

    _write_ascii_line(stream, message + "\n")


def _reject_json_constant(value):
    raise ProtocolError("Non-finite JSON number is not allowed: %s" % value)


def read_message(stream):
    try:
        line = stream.readline()
    except Exception as exc:
        raise ProtocolError("Could not read protocol message: %s" % exc)

    if line == b"" or line == "":
        raise EOFError("Detector protocol stream reached EOF")

    if isinstance(line, binary_type) and not isinstance(line, text_type):
        try:
            line = line.decode("ascii")
        except (UnicodeDecodeError, AttributeError) as exc:
            raise ProtocolError("Protocol message is not ASCII: %s" % exc)

    if not line.strip():
        raise ProtocolError("Protocol message is blank")

    try:
        payload = json.loads(line, parse_constant=_reject_json_constant)
    except ProtocolError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Malformed JSON protocol message: %s" % exc)

    if not isinstance(payload, dict):
        raise ProtocolError("Protocol message must be a JSON object")
    return payload


def _is_non_empty_text(value):
    return isinstance(value, string_types) and bool(value.strip())


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, integer_types + (float,)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return not math.isnan(number) and not math.isinf(number)


class DetectorClient(object):
    def __init__(self, request_stream, response_stream):
        self._request_stream = request_stream
        self._response_stream = response_stream
        self._next_request_id = 1
        self._poisoned = False

    def detect(self, image_path, target):
        if self._poisoned:
            raise ProtocolError("Detector client is poisoned and unusable")
        if not _is_non_empty_text(image_path):
            raise ProtocolError("image_path must be non-empty text")
        if not _is_non_empty_text(target):
            raise ProtocolError("target must be non-empty text")

        request_id = self._next_request_id
        self._next_request_id += 1

        try:
            write_message(
                self._request_stream,
                {"id": request_id, "image_path": image_path, "target": target},
            )
            response = read_message(self._response_stream)
            self._validate_response_identity(response, request_id)

            ok = response.get("ok")
            if not isinstance(ok, bool):
                raise ProtocolError("Response ok must be a boolean")
            if not ok:
                error = response.get("error")
                if not _is_non_empty_text(error):
                    raise ProtocolError(
                        "Failed response must contain a non-empty error"
                    )
            else:
                self._validate_success_response(response, target)
        except (EOFError, ProtocolError):
            self._poisoned = True
            raise

        # A matched, well-formed business error leaves the stream synchronized.
        if not ok:
            raise ProtocolError(error)

        return response

    @staticmethod
    def _validate_response_identity(response, request_id):
        response_id = response.get("id")
        if (
            isinstance(response_id, bool)
            or not isinstance(response_id, integer_types)
            or response_id != request_id
        ):
            raise ProtocolError("Response id does not match request id")

    @staticmethod
    def _validate_success_response(response, target):
        class_id = response.get("class_id")
        if (
            isinstance(class_id, bool)
            or not isinstance(class_id, integer_types)
            or class_id < 0
        ):
            raise ProtocolError("Response class_id must be a non-negative integer")

        if not _is_non_empty_text(response.get("class_name")):
            raise ProtocolError("Response class_name must be non-empty text")

        confidence = response.get("confidence")
        if (
            not _is_finite_number(confidence)
            or float(confidence) < 0.0
            or float(confidence) > 1.0
        ):
            raise ProtocolError("Response confidence must be finite and in [0, 1]")

        box = response.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ProtocolError("Response box must contain exactly four numbers")
        if not all(_is_finite_number(coordinate) for coordinate in box):
            raise ProtocolError("Response box coordinates must be finite numbers")
        x1, y1, x2, y2 = [float(coordinate) for coordinate in box]
        if x2 <= x1 or y2 <= y1:
            raise ProtocolError("Response box must have positive width and height")

        response_target = response.get("target")
        if not _is_non_empty_text(response_target) or response_target != target:
            raise ProtocolError("Response target does not match request target")
