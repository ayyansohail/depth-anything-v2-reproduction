# Hosted invalid-upload check

`invalid_image.png` deliberately contains plain text rather than PNG bytes. It is not an image and is expected to be rejected.

In the hosted app on Edge:

1. Start with the successful Small result visible, then replace the upload with this file.
2. Confirm the old depth map and download controls disappear, and a readable error says the image could not be read.
3. Replace it with the original valid living-room input photo. Confirm no stale depth map appears, click Generate depth, and confirm a new prediction succeeds.

Return the displayed error and whether recovery succeeded. Run these checks in the hosted app only; this fixture does not require local execution.
