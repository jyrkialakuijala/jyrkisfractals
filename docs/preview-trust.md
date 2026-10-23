# Preview trust: keeping black and color apart across zooms

A note to whoever works on the zoom preview next, including me. Companion to
`pending-samples.md`, which is about samples of one pixel; this is about what
an earlier image may say about a later one.

## What a preview is

When the view changes, the last displayed image is resampled (bicubic) into
the new view and shown until real samples arrive. It also keeps standing in,
after that, for pixels whose own samples have all hit the limit: the previous
view may have been deepened much further than the new one has been so far, and
replacing its colors by black would flicker. That hold is the "trusted preview"
rule, and it is where halos come from if it is applied carelessly.

## Classes in the alpha channel

Every displayed pixel carries its class in its alpha byte, and resampling
carries the class of the nearest source pixel (not an average: 255 and 253
averaged would read as 254, a different class):

| alpha | class   | meaning                                                   |
|-------|---------|-----------------------------------------------------------|
| 255   | trusted | every sample escaped, and no neighbour is black           |
| 254   | shallow | the surround fill, computed to a low limit only           |
| 253   | mixed   | some sample hit, or a neighbour is black: part black      |
| 0     | nothing | never rendered (the border of a zoom out before the fill) |
| other | partial | the resampled area was only partly rendered               |

Only a *trusted* color is ever held against a pixel's own hits (to the depth
the previous view verified); a *shallow* color only to the surround's depth; a
*mixed* color never. The reason: a mixed pixel's mean says "color" about an
area that is partly black, and after a zoom the pixels inside that area would
keep a color they never had, until the deepening caught up with the previous
view's depth, which with a deep previous view means for minutes. The ring of
color pixels adjacent to black is treated as mixed too, even if all its samples
escaped, because such a pixel may cover a sliver of black its samples missed.

## Black and color are resampled separately

The resampler decides by the nearest source pixel whether a result is black (a
rendered pixel with no color), and interpolates a non-black result's color from
the non-black taps only. So a magnified black region keeps its boundary instead
of being fringed with smeared boundary colors, and the boundary colors are not
darkened by black taps either. Before this, bicubic interpolation alone put
about 10000 colored pixels inside the black areas of a 4K view after four
quick zooms, and the trusted rule then held them to the end.

## Depth is compared per pixel

Whether the preview is "deeper" than a pixel is decided against the depth that
pixel's own samples have reached (`Accum::depth`), not against the pass limit.
With the planner, a zoom in may start at a deep limit at once, and the samples
then catch up over several budgeted passes; comparing with the pass limit
would drop the trust in the first pass, with nothing verified yet.

## One sample does not refute eight

A black pixel under trusted color keeps that color while it has fewer than its
full set of samples, even when its one sample hit as deep as the previous view
went: the previous pixel's eight samples escaped, so this pixel is a boundary
sliver at best, and its antialiasing samples settle it. Dropping the color
there flickered a few dozen boundary pixels black for ten frames on every zoom.

## The tests

- `--test-halo F N [--size 3840x2160]`: converges a view, zooms in N times with
  one pass between zooms, then watches every frame of the final view. A pixel
  black by rule at the end and not adjacent to color must never show color in
  a rendered frame. The preview frame itself is reported, not judged, and the
  one-pixel band next to color is tolerated.
- `--test-flicker F`: the opposite direction. A pixel that showed trusted color
  in the preview and is colored at the end must never be black by rule in
  between. Mixed and shallow pixels that go black are reported with how long
  they stayed black; they are expected (black and color stay separate).

Both must pass after any change to the preview, the resampler, the depth
bookkeeping or the sample schedule.
