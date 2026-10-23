# JyrkisFractals

Interactive GPU fractal viewer. CUDA computes every frame directly into an OpenGL
pixel buffer; SDL2 provides the window and input. Currently renders the Mandelbrot set.

## Build

The machine has no system-wide CUDA toolkit and no root access, so the compiler and
runtime come from NVIDIA's pip wheels (CUDA 13.4) in a user-local venv, and the SDL2 /
OpenGL headers are vendored under `third_party/include`. Run once:

    ./setup_toolchain.sh
    make

If you have a regular CUDA install and the `-dev` packages, override the defaults:

    make CUDA_HOME=/usr/local/cuda ARCH=sm_120 SDL_LIBS=-lSDL2 GL_LIBS=-lGL

`ARCH` defaults to `sm_120` (RTX 5090 / Blackwell).

## Run

    ./jyrkisfractals                      # interactive window
    ./jyrkisfractals --size 2560x1440
    ./jyrkisfractals --center -0.7436438870371 0.1318259042053 --zoom 2e9
    ./jyrkisfractals --headless out.png --size 3840x2160 --iter 5000   # no window
    ./jyrkisfractals --aa 16                                             # 16 samples per pixel (1, 2, 4, 8 or 16)
    ./jyrkisfractals --bench 200 --size 1920x1080                      # kernel timing
    ./jyrkisfractals --help

| Input              | Action                                            |
|--------------------|---------------------------------------------------|
| mouse wheel        | zoom around the cursor                            |
| left drag          | pan                                               |
| double click       | zoom in 2x at cursor (right click: zoom out 2x)   |
| Z / X              | zoom in / out around the center                   |
| PgUp / PgDn        | zoom in / out around the mouse pointer            |
| arrows             | pan by a tenth of the window                      |
| 1 .. 9, 0          | antialiasing: samples per pixel, rounded up to 1, 2, 4, 8 or 16 (0 = 16, default 8)|
| + / -              | more / fewer iterations (turns off auto)          |
| A                  | toggle automatic iteration count                  |
| D                  | precision: auto / float / double                  |
| P / Shift+P        | next / previous palette                           |
| [ / ]  , / .       | stretch / rotate the palette                      |
| S                  | save fully antialiased PNG (prints coordinates)   |
| F                  | fullscreen                                        |
| R                  | reset view                                        |
| H                  | print help; Esc or Q quits                        |

The window title shows zoom, iteration count, precision in use, antialiasing progress, and
kernel time.

## How it works

- `src/kernels.cu` holds the templated escape-time kernel. One instantiation runs in
  `float`, one in `double`; `Auto` precision switches to double once a pixel is smaller
  than 1e-6 complex units (zoom of roughly 2000x on a 1080p window), which is where float
  visibly breaks. Points inside the main cardioid and period-2 bulb are skipped. Coloring
  uses the continuous (smooth) escape time with a large bailout radius.
- `src/main.cpp` owns the SDL2 window, a GL 2.1 context, and a pixel buffer object that is
  registered with CUDA. Each frame maps the buffer, launches the kernel, unmaps, and
  uploads to a texture without touching host memory. If interop registration fails the
  viewer falls back to a device-to-host copy automatically.
- Antialiasing is progressive supersampling with 1, 2, 4, 8 or 16 samples per pixel (keys
  1..9 round up to the next power of two, 0 is 16; default 8). A pixel's color is the exact
  mean of its samples, black for the ones that ran to the limit, and since the counts are
  powers of two the division is exact in floating point. The first pass takes two samples
  per pixel (the second only if the first escaped) and is shown immediately when the view
  moves. The next passes double the count early, to 4, 8 and 16, at the current low limit,
  for pixels whose samples have all escaped: cheap antialiasing for the bulk of the exterior
  right away. A sample that arrives later than the pixel's others and has not been taken
  as deep as they were resolved to (it ran out of the per-pass budget, or hit a lower limit
  than they were probed under) is pending: it is resumed every pass but adds no black to
  the mean until it has caught up, so a pixel never darkens while its extra samples are
  still on their way (`docs/pending-samples.md` explains the rule in full). A pass stops at a pixel's first sample that hits the limit, so a pixel with
  unresolved hits can hold another count for a while; its hits are re-probed and replaced as
  the limit deepens, and the count reaches the power of two once they resolve or its
  remaining samples are taken. A pixel whose extra sample hits that limit drops back to its two primary
  samples alone (the extra ones are ignored, the primary ones never touched) and gets its
  remaining samples after the depth phase instead. Depth comes before antialiasing for those: the deepening described below
  runs on their single sample until it has settled (nothing newly exposed and
  fewer than 5 pixels rescued by a deepening pass, or the limit has caught up with a trusted
  preview), and only then do the remaining samples accumulate in a float buffer, doubling
  the count per pass at that depth, within an 8 ms kernel budget per displayed frame; the deepening
  resumes afterwards. A pixel whose four neighbours are black keeps its single sample and
  gets no further subpixels, since those would almost surely run to the limit too. The sample pattern is not a grid: the first sample is the pixel
  center and every next one lands in the biggest gap left in the pixel square (measured
  toroidally, so the pattern tiles across neighbouring pixels), nudged by a random amount of
  up to 0.3 of that gap's radius so it stays a little chaotic. Partial results are therefore
  always evenly spread over the pixel. `--dump-samples --aa N` prints the pattern.
  Screenshots wait for all samples.
- Zooming shows a preview first: the last displayed image is resampled with bicubic
  (Catmull-Rom) interpolation into the new view and shown immediately, then real samples
  replace it pixel by pixel. Black and non-black areas are resampled separately: the nearest
  source pixel decides whether a pixel is black, and colors are interpolated from colored
  taps only, so a magnified black region keeps its boundary instead of a fringe of smeared
  colors (`docs/preview-trust.md` explains the whole scheme). The preview frame is only spent when a full pass costs more
  than 3 ms, so cheap views still get the real image on the first frame. Every pixel carries
  a weight in its alpha channel: an area nobody has rendered yet has weight 0, resamples to
  weight 0, and adds nothing when real samples arrive, no matter how many zoom events are
  stacked on top of each other. The border a zoom out or a pan uncovers is not left black
  until its first pass, though: the viewer keeps a surround image, a half-resolution render
  of a field four times wider than the view at a low limit (1000 iterations, or the auto
  limit of that field if higher), redone whenever the view leaves the field it covers (it
  costs a fraction of a millisecond in single precision), and the preview's empty pixels are
  filled from it. Those pixels get alpha 254 instead of 255: the fill is a prior, but its
  colors are only trusted to the surround's depth, where the resampled image's colors are
  trusted to the depth the previous view reached (the alpha classes carry this through any
  number of further zooms). So a sequence of zoom-outs never flickers to black in the ring
  the view just uncovered.
- Interior pixels are cheap and get deepened instead of antialiased. Each pixel keeps its
  color sum and sample count: an escaped sample adds its color, a sample that ran to the
  iteration limit adds black, and the pixel shows the plain mean. Only a pixel whose samples
  all hit the limit is pure black; a hit sample that escapes at a deeper limit later is
  replaced by its color: whenever a pass had limit-hit pixels, the effective limit for the
  next pass grows by 500, so "false interior" pixels that merely needed more iterations get
  resolved while the view sits still. Each pixel remembers which of its samples hit the
  limit, and once all its samples are taken those exact samples are re-probed at the deeper
  limit every pass: a hit again changes nothing, an escape replaces the black sample with
  its color. The orbit (z, iterations done) of every hit sample is stored per pixel and
  sample, so each deeper probe resumes where the previous one stopped and costs only the 500
  new iterations, whatever the limit; resuming is bit-exact. Each sample slot also keeps the
  smooth escape value of an escaped sample, so palette, color scale and color offset changes
  (`P`, `[`, `]`, `,`, `.`) only recolor the kept samples instead of re-rendering. That
  storage is 16 bytes per sample per pixel in single precision (24 in double) plus 20 bytes
  of bookkeeping per pixel, all of a pixel's samples contiguous: about 150 MB at 1280x800
  with 8 samples (220 MB in double), small enough to be cache friendly for the boundary
  pixels a pass touches. If it cannot be allocated the viewer says so and exits. Each pass counts how many probed pixels ran to the limit (the title shows it as
  `hits 37`), and passes keep going, raising the limit by 500 each, until fewer than 5 pixels
  hit or the limit reaches 4194304. The limit is held steady in any pass that re-probed
  pixels the eroding black region had just exposed (their stored orbits lag behind the
  previous pass), so the erosion catches up at the current depth before deepening goes on. Pixels at the edge of true interior hit at every limit,
  so a view with real interior on screen keeps deepening until that cap. The adjustment
  resets to 0 on every zoom and the title shows it as `iter 3438+3000`. The zoom preview prior counts as 8 samples
  at first and fades out entirely by the time all samples are in, so nothing of
  the magnified old image lingers, but a pixel whose samples all hit the limit is shown as
  pure black whatever the preview says, with one exception: the preview remembers the
  depth its colors were tested to (the limit the previous view reached), and where that is
  higher than the current pass's limit and the preview shows color, the pixel is not taken
  as black: the color stays until the pixel's own samples have reached that depth or one
  escapes (unless the coarse image described below blocks the pixel, which clears it to
  black). Only colors of pixels whose samples all escaped, and that have no black neighbour,
  are trusted that way; the color of a boundary pixel that was part black (some samples hit,
  or next to black) is a prior but is never held against a hit, since the claim is false for
  its black part and holding it would fringe every magnified black area with a halo. A black
  pixel under trusted color also keeps it while it has fewer than its full samples: one hit
  sample does not refute a whole pixel's mean. Those colors are trusted through successive
  zooms until they have been magnified by more than 5x in total; beyond that the black
  areas are computed from scratch. Separately, a zoom in
  starts the new view at the depth the image on screen needed (the limit of the last pass
  that still rescued a noticeable share of the probed pixels, at least 5 or 1% of them, not
  the cap the deepening may chase afterwards), so nothing resets to a shallower limit. A
  zoom out starts at the auto limit, so the uncovered border is drawn quickly. Pixels inside the main cardioid or period-2
  bulb are interior for certain and end up black, but on the boundary of the cardioid a
  pixel whose first sample lies inside may still have escaping samples, so under a trusted
  deeper preview it shows the preview's antialiased color like any other black pixel until
  its samples are in (otherwise every zoom would draw a black fringe along the cardioid for
  a frame). Iteration keys (`+`, `-`, `A`) set the
  limit relative to the current effective one. A black pixel under trusted preview color keeps the preview
  color, the tile census does not count it, and it is probed again every pass. The
  preview's black is also eroded by one tile for the skip rule: a black pixel whose nearest
  2x2 block of tiles contains any color of the deeper preview is never skipped, so the band
  along the previous view's boundary is probed deeper after every zoom. Deepening does not
  stop before the limit has reached the preview's, so every such pixel ends up either
  confirmed by an escape or replaced by black; with resumed orbits that costs little.
  Headless renders use the given limit without deepening.
- A zoom plans its first pass with the coarse field. Before the first full-resolution pass of
  a zoomed view, the 4x4 coarse image alone probes the view, in stages that double the limit
  from the auto limit up to the depth the previous image needed (a zoom out probes the auto
  limit only), until the probing has spent 10 ms of kernel time (it usually takes a fraction
  of a millisecond; about 2 ms at deep double-precision zooms). Each stage predicts the cost
  of a first pass at that depth from the field: the 16 pixels of an escaped point cost about
  what the point took, per sample; pixels around a boundary point (one with an escaped
  neighbour) one sample at the limit, capped by the per-pass budget; blocked interior pixels
  nothing. The deepest stage that fits 30 ms with the two primary samples sets the starting
  limit (failing that, the auto limit with one sample), and the sample count then doubles,
  up to the full count, while the prediction still fits. A zoom in may start up to four
  times deeper than the depth the previous image needed, never deeper than it verified, so
  boundary pixels that need a little more depth catch up through the per-pass budget in a
  few passes instead of 500 iterations per pass. The rate behind the prediction is
  what the last first pass really achieved in that precision, learned from its measured
  time, so predictions settle within a few percent after the first zoom (the probe's own
  rate, which runs far fewer threads, stands in before that and errs on the slow side). The
  probed field stays for the passes, which block with it from the start. `JF_TRACE=1`
  prints the stages and the plan.
- Every pass is bounded. A sample iterates at most 4096 iterations in one pass; one that has
  neither escaped nor reached the limit by then stores its orbit as in progress, counts as
  black for the time being and as newly exposed, so the deepening waits, and continues next
  pass. The first pass after a zoom to an inherited depth of 40000 is thus ten short passes
  instead of one long one: frames keep coming, and no kernel launch can run long enough to
  trip the display driver's watchdog. The budget also adapts to real time: when an update
  (the passes of a frame plus resolving and displaying it) took more than 100 ms, the budget
  of the next passes is cut in proportion, aiming at 50 ms, down to 64 iterations, so the
  next update does not take that long again; updates under 25 ms double it back, up to
  4096. The title shows `(budget cut)` while it is lowered.
- A coarse image guards the interior. Every pass first iterates one point per 4x4 tile (the
  tile center), resumed from its stored orbit like the pixels' samples, so it costs next to
  nothing. Its budget is four times the pixels' per pass. After a zoom out or a pan it runs
  towards the deeper of the pass's limit and the depth a trusted preview's colors were tested
  to, so it can be ahead of the pixels; after a zoom in it starts clean and follows the new
  view's limit only, so no computation towards the previous view's depth carries over (the
  preview's colors then give way to the pixels' own deepening). A pixel whose 2x2 nearest coarse points have all been taken without
  escaping at least as deep as the pixel's own depth, the limit it probes to, or, where it
  shows a trusted preview's color, the depth that color was tested to, is taken as black
  without being computed at all, not even its first sample, and shown as pure black whatever
  the magnified preview had there: no interpolated halo of an earlier rendering lingers,
  and the blocker never clears a color that was verified deeper than its own points. The
  color comes back only if a coarse point escapes at a deeper limit, and a pixel that already
  has an escaped sample is never overridden. On a pan the coarse image is shifted along when
  the pan is a whole number of tiles and restarted otherwise.
- Deepening only probes the edge of black regions. Each pass first counts the black pixels in
  every 4x4 tile, where a pixel is black only if every real sample it has taken hit the limit
  (pixels the rule stopped probing keep that state; pixels showing only the zoom preview have
  no samples and never count). A black pixel whose own tile and the three neighbouring tiles
  on its side (the nearest 2x2 block of tiles) are completely black is left as is.
  Black areas are thus eroded from their boundary, a tile row or so per pass, and the bulk of
  a bulb or mini-brot costs nothing. `JF_NO_TILE_SKIP=1` in the environment disables the rule
  for comparison; `--save-on-exit F.png` with `--exit-after` writes the converged GUI view;
  `JF_TRACE=1` prints per-pass limit, hit and rescue counts to stderr; `--auto-zoom N` zooms
  in 1.25x at the center N times, one every 400 ms, for scripted runs.
- Nothing flashes black. `--test-flicker F` converges a view the way the GUI does, zooms by
  `F` (below 1: out) about the center, and checks every frame until the new view has
  converged too: a pixel the preview showed colored and that is colored at the end must never
  have been black by rule in between (all samples hit, or blocked by the coarse image; an
  exact mean of near-black palette colors can round to zero and does not count). Colors the preview held as trusted may not flash at all; the
  surround fill is approximate, so up to 0.5% of its pixels may (a pixel colored in the
  surround at 1000 iterations can hit a deeper limit and be rescued deeper still).
  `JF_FLICKER_DUMP=1` lists some of the offending pixels, `JF_FLICKER_PIXEL="x y"` traces one
  through the first passes. Mixed and shallow pixels that go black are reported with how long
  they stayed black, not judged: black and color are kept separate on purpose.
- Nothing haloes either. `--test-halo F N` (use `--size 3840x2160`, where the effect was
  first seen) converges a view, zooms in N times quickly with one pass between zooms, then
  watches every frame of the final view: a pixel that is black by rule at the end and not
  adjacent to color must never have shown color in a rendered frame. The preview frame
  itself is reported, not judged, and the one-pixel band next to color is tolerated.
- Panning reuses work. The accumulation buffer stores a per-pixel sample count, drags are
  snapped to whole pixels, and a pan translates the buffer on the GPU so only the newly
  exposed strip needs computing; pixels still on screen keep all their samples.
  `--test-shift DX DY` checks this against a direct render.
- Frames are rendered only when the view changes or samples remain; the loop otherwise
  sleeps on events.
- `src/image_io.cpp` writes PNGs through zlib.

Measured on an RTX 5090 at 1920x1080 with 16 samples: the default view at 256 iterations
takes about 2 ms of kernel time for all samples; a 2e9x zoom at ~3400 iterations in double
precision takes about 25 ms per sample, so the first sample appears immediately and the rest
arrive over a fraction of a second (double throughput on consumer GPUs is 1/64 of float).
The default of 8 samples halves both the time and the storage.

## Next steps

- Perturbation theory with a high-precision reference orbit for zooms beyond ~1e13x,
  where double precision runs out.
- More fractals: Julia sets (same kernel with fixed `c`), Burning Ship, Multibrot. The
  kernel is templated so these are mostly new inner loops plus a `FractalType` entry.
- Adaptive antialiasing: spend extra samples only on pixels whose neighbours differ.
- Low-resolution preview while dragging once iteration counts get high enough for a
  single sample to take longer than a vsync.
