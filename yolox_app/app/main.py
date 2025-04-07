import actfw_core
import consts
from actfw_core.capture import V4LCameraCapture
from actfw_core.system import find_usb_camera_device, get_actcast_firmware_type
from actfw_core.unicam_isp_capture import UnicamIspCapture
from tasks import Drawer, Predictor, Preprocessor, Presenter


def main():
    # Actcast application
    app = actfw_core.Application()

    # Load act setting
    settings = app.get_settings({
        'display': True,
        "camera_mode": "module",
        'thresh': 0.35
    })
    cmd = actfw_core.CommandServer()
    app.register_task(cmd)

    camera_mode = settings["camera_mode"]
    if camera_mode == "module":
        if get_actcast_firmware_type() == "raspberrypi-bullseye":
            # Use UnicamIspCapture on Raspberry Pi OS Bullseye
            cap = UnicamIspCapture(unicam="/dev/video0", size=(consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT), framerate=30)
        else:
            cap = V4LCameraCapture("/dev/video0", (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT), 30)
        actual_cap_size = cap.capture_size()
    elif camera_mode == "usb":
        device = find_usb_camera_device()
        cap = V4LCameraCapture(device, (consts.CAPTURE_WIDTH, consts.CAPTURE_HEIGHT), 30)
        actual_cap_size = cap.capture_size()
    app.register_task(cap)

    # Predictor task
    preproc = Preprocessor(actual_cap_size)
    app.register_task(preproc)
    pred = Predictor(settings['thresh'])
    app.register_task(pred)

    draw = Drawer(settings)
    app.register_task(draw)

    pres = Presenter(
        cmd,
        use_display=settings['display']
    )
    app.register_task(pres)
    cap.connect(preproc)
    preproc.connect(pred)
    pred.connect(draw)
    draw.connect(pres)
    app.run()


if __name__ == "__main__":
    main()
