# Your doodles go here

The background already draws its own animated doodles. Anything you add here
floats on top of them.

1. Pick SVG or PNG doodles from a free pack, for example
   - Open Doodles: https://www.opendoodles.com (CC0)
   - unDraw: https://undraw.co (free licence)
   - Doodle Ipsum / Storyset line art
2. Drop the files into this folder (`frontend/public/doodles/`).
3. List them in `manifest.json`:

```json
[
  { "file": "reading.svg",  "x": 4,  "y": 62, "size": 180, "rotate": -6 },
  { "file": "idea.png",     "x": 84, "y": 18, "size": 140, "rotate": 8 }
]
```

`x` and `y` are percentages of the screen (left, top). `size` is the width in
pixels. `rotate` is degrees. Reload the page; no rebuild needed.
