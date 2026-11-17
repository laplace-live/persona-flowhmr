from html.parser import HTMLParser

from persona_flowhmr.web_ui import _DASHBOARD_HTML


def test_full_video_controls_are_opt_in_and_offer_paired_and_individual_downloads():
    controls = {}
    formats = set()

    class ControlsParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                controls[attrs["id"]] = attrs
            if tag == "option" and attrs.get("value", "").endswith((".zip", ".vmd", ".vrma", ".fbx")):
                formats.add(attrs["value"])

    ControlsParser().feed(_DASHBOARD_HTML)
    auto = controls["autoRecordVideo"]
    assert auto["type"] == "checkbox" and "checked" not in auto and "data-cfg" not in auto
    assert controls["recordVideo"]["onclick"] == "recordFullVideo()"
    assert controls["downloadTakeBtn"]["onclick"] == "downloadTake()"
    assert "downloadTake" not in controls
    assert formats == {
        "avatar-vrma-camera.zip", "avatar-fbx-camera.zip", "avatar-vmd-camera.zip",
        "avatar.vrma", "avatar.fbx", "avatar.vmd", "camera.vmd",
    }


def test_dashboard_downloads_go_directly_to_the_attachment_endpoint():
    assert "location.replace('/api/recording/download/'" in _DASHBOARD_HTML
    assert ".blob()" not in _DASHBOARD_HTML
    assert "createObjectURL" not in _DASHBOARD_HTML
