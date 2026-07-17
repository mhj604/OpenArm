# Rollback

This setup is isolated. To disable it without deleting anything:

1. Stop any process launched from `/home/mh/openarm_act/scripts` with `Ctrl-C`.
2. Remove `/home/mh/openarm_act` from the current shell `PATH` by closing that shell or starting a new one.
3. Do not source `/home/mh/openarm_act/scripts/env.sh`.

To remove the RealSense system permission rule, authenticate once in the desktop
prompt and run:

```bash
pkexec /home/mh/openarm_act/scripts/install_realsense_udev.sh uninstall
```

The helper removes the rule only when its checksum matches the official file it
installed. Starting a new shell without sourcing `scripts/env.sh` restores the
original pyrealsense2 import path automatically. The original Python wheel is
also retained under `/home/mh/openarm_act/backups/realsense_20260716_154035`.

To remove the official DKMS driver package and return module selection to the
Ubuntu kernel package:

```bash
sudo apt-get remove librealsense2-dkms
sudo depmod -a
sudo reboot
```

Pre-install copies of `uvcvideo.ko` and `videodev.ko`, plus their metadata, are
stored in `/home/mh/openarm_act/backups/realsense_dkms_20260716`. Package
removal is preferred over manually copying those modules back.

If deletion is later requested, remove the udev rule first and then remove only
`/home/mh/openarm_act`. Existing OpenArm, ExoArm, ROS2, CAN, and camera firmware
files do not need restoration.

`/home/mh/openarm_imitation` is a separate earlier isolated directory and should not be deleted as part of this rollback unless the user explicitly requests it.
