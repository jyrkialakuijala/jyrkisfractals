# Pending samples: why a late subsample must not darken a pixel

A note to whoever works on `src/kernels.cu` next, including me.

## The problem

A pixel's color is the plain mean of its samples: an escaped sample adds its
color, a sample that ran to the iteration limit adds black. The samples do not
arrive together. Two primary samples come in the first pass, the rest come
later: early, at a low limit, for pixels whose samples have all escaped, and
after the depth phase, at the settled limit, for everyone else. Each sample is
also bounded to 4096 iterations per pass (less when frames run long), so a
sample that needs 40000 iterations to reach the limit is "unfinished" for ten
passes and counts as a hit meanwhile.

So a pixel whose two primary samples escaped under a limit of 40000 receives a
third sample that, after one pass, has done 4096 iterations and has not escaped
yet. If that sample's black were mixed into the mean, the pixel would darken by
a third for several frames and brighten again when the sample finally escapes.
The same happens whenever a new sample is probed to a lower depth than the
pixel's earlier samples were: its black is not comparable to their colors. That
is the flicker to black during subsampling.

## The rule

Every pixel keeps `depth`: how deep its samples are resolved to. An escaped
sample resolves the pixel to the limit of the pass it escaped under (it would
have gone on to that limit otherwise); a sample that hit the limit, or ran out
of budget, resolves it to the iteration count it reached. `depth` is the
maximum over the pixel's samples, and only grows.

A sample that hit (or is unfinished) **counts as black only if the iteration
count it reached is at least `depth`**. Otherwise it is *pending*: its hit bit
stays set, so it is resumed from its stored orbit every pass like any hit, but
it adds nothing to the color sum and nothing to the weight. The pixel's mean is
therefore taken over the samples that are comparable in depth, and a pending
sample joins it only once it has caught up: either it escapes, and adds its
color, or it reaches the pixel's depth, and adds black that is as well founded
as the rest of the pixel.

Since all hit samples of a pixel are re-probed to the same limit every pass,
they stay in step except for budget lag, and a lagging sample catches up at
4096 iterations per pass against a limit that grows by 500. Pending samples
are counted as hits in the per-pixel counters, so a pixel whose samples are all
hits or pending is still "black" for display and for the tile census; the rule
only changes how black mixes into a pixel that has color.

## Where it lives

- `Accum::depth` (`src/fractal.h`): one `int` per pixel, cleared with the
  samples, shifted with the pixel on a pan.
- `hitCounts(reached, depth)` and `replayPixel(...)` in `src/kernels.cu`: the
  single place that decides a hit's weight. The live accumulation in the
  sequential sample loop applies the same test with the depth the pixel had
  before the sample; every re-probe replays the whole pixel afterwards, so a
  sample that just caught up is counted from then on, and recoloring replays
  with the same rule, so all paths stay bit-identical.
- The early phase keeps its own stronger rule on top: an extra sample that hits
  the low limit drops all extra samples and the pixel falls back to its two
  primary samples (`kEarlyFailed`), because at that point the new sample is as
  deep as the primaries and would legitimately count.

## What it is not

It is not the "deeper preview" rule, which keeps a magnified previous image's
color for a pixel whose own samples are all black. That compares the pixel to
another image (see `preview-trust.md`); this compares a pixel's samples to each
other. The two share `Accum::depth`: the preview rule compares its depth with
the depth the pixel's samples reached, not with the pass limit.
