from pathlib import Path


LAUNCH = (
    Path(__file__).resolve().parents[2]
    / "mirobot_ws/src/astra_camera/launch/astrapro.launch"
)


def test_rgb_camera_node_uses_launch_camera_info_url_argument():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '<param name="camera_info_url" value="$(arg rgb_camera_info_url)"/>' in source
    assert '<param name="camera_info_url" value=""/>' not in source
