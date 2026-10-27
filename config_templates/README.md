
# Config Template

This directory contains the default configuration files. One of the configuration file, `conf.default.yaml` by default, will be copied to the root directory of this project as `conf.yaml`.

Do not modify the content of these template unless you are contributing to this project.

## Shared-screen input without camera or system audio

To forward images from a user-selected screen share while blocking camera images
and system audio, set these values under `system_config` in your local `conf.yaml`:

```yaml
enable_screen_capture: false
enable_screen_sharing: true
```

`enable_screen_sharing` defaults to `false`, preserving existing behavior when
omitted. Explicit image uploads remain allowed. With `enable_screen_capture`
disabled, only images labeled `source: upload`, and `source: screen` when screen
sharing is enabled, reach the conversation queue. Camera images, unknown image
sources, and system audio remain blocked. This filters the sources reported by
the client; it does not authenticate the origin of image pixels. Enabling
`enable_screen_capture` retains the legacy behavior of allowing all captured
images and system audio. Restart OLV safely after changing these settings.
