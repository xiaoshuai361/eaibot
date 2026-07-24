from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELAY = ROOT / "src/tag_yolo_quiet_zone_relay.py"


def test_yolo_relay_log_names_cached_boxes_and_stays_out_of_info_stream():
    source = RELAY.read_text()

    assert "YOLO quiet-zone relay publishing; boxes=%d refresh=%s" not in source
    assert "rospy.loginfo_throttle(2.0, 'YOLO quiet-zone relay publishing" not in source
    assert "cached_yolo_boxes=%d refresh_yolo=%s" in source
    assert "rospy.logdebug_throttle" in source
