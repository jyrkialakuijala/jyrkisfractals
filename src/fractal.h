// Shared definitions between the CUDA kernels and the host application.
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

enum class FractalType : int { Mandelbrot = 0 };
enum class Precision : int { Auto = 0, Float = 1, Double = 2 };

constexpr int kMaxSamples = 16;  // samples per pixel at most (bounds the per-pixel storage)

// Everything that determines what a frame looks like.
struct ViewParams {
  double centerX = -0.75;
  double centerY = 0.0;
  double scale = 3.5 / 1280.0;  // complex-plane units per pixel
  int maxIter = 256;
  int palette = 0;
  float colorScale = 0.02f;   // palette cycles per iteration
  float colorOffset = 0.0f;   // phase shift of the palette, in cycles
  int samples = 8;            // antialiasing: samples per pixel, rounded up to a power of two <= kMaxSamples
  FractalType fractal = FractalType::Mandelbrot;
  Precision precision = Precision::Auto;
};

struct RenderStats {
  bool usedDouble = false;
  bool hitLimit = false;  // some pixel ran all the way to maxIter in this pass
  int hitCount = 0;       // how many probed pixels ran all the way to maxIter in this pass
  int rescueCount = 0;    // how many re-probed pixels had a sample escape in this pass
  int maxRescueIter = 0;  // the highest iteration count at which a rescued sample escaped
  int exposedCount = 0;   // pixels with a sample lagging behind settledIter or still in progress (see kIterBudget)
  float kernelMs = 0.0f;
};

constexpr int kPaletteCount = 6;

// Below this pixel size single precision visibly breaks down, so Auto switches to double.
constexpr double kDoubleThreshold = 1e-6;

// Number of samples per pixel for this view: p.samples rounded up to a power of two (1, 2, 4, 8
// or 16), so that the pixel averages are exact divisions.
int totalSamples(const ViewParams& p);

// The subpixel sample pattern for `n` samples (offsets in [-0.5, 0.5) about the pixel center,
// in the order the samples are taken): a greedy largest-gap sequence with a little jitter.
void samplePattern(int n, std::vector<float2>& out);

// The accumulation buffer holds, per pixel, the color sum in color.xyz and its total weight
// in color.w, plus two small counters in meta: the number of samples taken so far and how
// many of them ran to the iteration limit (and a "certain interior" bit, see isCertain).
// Row 0 is the top of the image.
//
// Adding a sample multiplies the existing sum and weight by kSampleAging (1: no aging) and
// adds the sample: an escaped sample adds its color, a limit-hit sample adds black, both at
// weight 1 (kLimitHitWeight), so a pixel's color is the exact mean of its samples with black
// for the ones that hit. The app takes samples in counts of 2, 4, 8 and 16, so the mean is a
// division by a power of two, exact in floating point. A hit sample that escapes at a deeper
// limit later is replaced by its color (replayPixel). A pixel is "black" when every sample it
// has taken hit the limit.
//
// A sample's black only counts once it has been taken as deep as the pixel's other samples
// were resolved to (depth[]): a later sample that ran out of per-pass budget, or hit a lower
// limit than the pixel's earlier samples were probed under, is "pending": it keeps its hit
// bit (so it is resumed every pass) but adds nothing to the sum or the weight until it has
// caught up, so the pixel's color does not darken meanwhile. See docs/pending-samples.md.
//
// hitMask remembers which of the pixel's samples hit the limit. Once a pixel has all its
// samples, every pass re-probes one of those samples at the current (deeper) limit: a hit
// again changes nothing, an escape replaces the black sample with the color. So passes keep
// going, and the limit keeps rising, as long as pixels hit the limit (the app stops when
// fewer than kFewHits pixels do).
//
// orbit / orbitIters keep, per pixel and sample, the iteration state (z, iterations done) of
// samples that hit the limit, so re-probing them at a deeper limit resumes there instead of
// iterating from z = 0 again: deepening by 500 iterations costs 500 iterations per sample,
// whatever the limit. Resuming is bit-exact, since the orbit is the same sequence of
// operations either way. z is stored in the precision the view renders in (float2 or
// double2, see orbitDouble); mu keeps the smooth escape value of escaped samples so a
// recoloring needs no iteration. Everything is laid out per pixel (all of a pixel's slots are
// contiguous) and kept as small as possible so the working set stays cache friendly:
// 20 bytes per pixel plus 16 (float) or 24 (double) per sample.
struct Accum {
  float4* color = nullptr;
  unsigned short* meta = nullptr;    // taken << 8 | hits (both saturate at 127), bit 7: certain interior
  unsigned short* hitMask = nullptr;  // bit k set: sample k of the pattern hit the limit (or is pending)
  int* depth = nullptr;              // [pixel]: the depth its samples are resolved to (see docs/pending-samples.md)
  void* orbit = nullptr;             // [pixel * orbitSlots + slot]: z after orbitIters iterations (float2/double2)
  int* orbitIters = nullptr;         // [pixel * orbitSlots + slot]: iterations done (-1: free, INT_MAX: interior)
  float* mu = nullptr;               // [pixel * orbitSlots + slot]: smooth escape value of an escaped sample
  int orbitSlots = 0;                // one per sample of the pattern
  bool orbitDouble = false;          // orbit holds double2 (else float2)
  // The coarse image: one point per 4x4 tile (its center), iterated every pass with a
  // resumable orbit like the pixels' samples, towards the deeper of the pass's limit and a
  // trusted preview's depth (kCoarseBudgetFactor times the pixels' budget per pass). Where its 2x2 points
  // nearest to a pixel have all been taken at least as deep as that pixel claims (its limit,
  // or the preview's depth where it shows a deeper preview's color) without escaping, the
  // pixel is taken as interior, cleared to black and not computed (see isCoarseBlack).
  int tw = 0, th = 0;                // tiles per axis
  void* coarseOrbit = nullptr;       // [tile]: z after coarseIters iterations (float2/double2)
  int* coarseIters = nullptr;        // [tile]: iterations done (-1: start over, <= kCoarseEscaped: escaped, see below)
  unsigned char* coarseHit = nullptr;  // [tile]: 1 if the point has not escaped within coarseIters iterations
};
constexpr float kLimitHitWeight = 1.0f;
constexpr float kSampleAging = 1.0f;
// A sample iterates at most this many iterations in one pass. One that has neither escaped
// nor reached the limit by then stores its orbit as in progress, counts as a hit (black) for
// the time being and as newly exposed (so the deepening waits), and continues next pass. This
// bounds the work of every pass, however deep the limit, so frames keep coming and no kernel
// launch can run long enough to trip a watchdog. This is the default and maximum; the app
// lowers the budget of the next passes when a frame took too long in real time (renderPass's
// iterBudget).
constexpr int kIterBudget = 4096;
// The coarse 4x4 image's budget per pass is this many times the pixels' (see
// Accum::coarseOrbit). It has one point per 16 pixels, so that is still a fraction of the
// pixels' worst-case work; it lets the coarse image run ahead of the pixels, to the depth a
// trusted preview's colors were tested to.
constexpr int kCoarseBudgetFactor = 4;
// A coarse point that escaped at iteration i stores kCoarseEscaped - i in coarseIters: never
// iterated again, and the escape iteration stays available for the cost model of coarsePass.
constexpr int kCoarseEscaped = -2;

// Statistics of the coarse field after coarsePass: see that function.
struct CoarseStats {
  float kernelMs = 0;        // time of this call
  int points = 0;            // coarse points in all
  int hits = 0;              // points that have not escaped (to the depth they were taken)
  int boundary = 0;          // of those, points with an escaped neighbour
  double escapedIters = 0;   // iterations the escaped points took, in all
  double work = 0;           // iterations done by every point so far, in all
  bool complete = false;     // every point has either escaped or reached `target`
};

// The alpha channel of a preview / display pixel is its weight (0: nothing rendered there) and
// says how far its colors are trusted: kPreviewAlphaTrusted marks colors tested to the previous
// view's depth (previewIter): every sample of the pixel escaped. kPreviewAlphaShallow marks
// colors from the low-iteration surround image that fills in the border of a zoom out (tested
// only to shallowIter). kPreviewAlphaMixed marks the color of a pixel some of whose samples hit
// the limit: a boundary pixel, part black. Its color is a prior like the others, but it is not
// held against a hit after a zoom, since the claim "color to that depth" is false for its black
// part (holding it would fringe every magnified black area with a halo). Resampling keeps
// these classes where an area is fully covered and lowers alpha where it is mixed with nothing.
constexpr int kPreviewAlphaTrusted = 255;
constexpr int kPreviewAlphaShallow = 254;
constexpr int kPreviewAlphaMixed = 253;
// Depths a preview's colors were tested to, by alpha class (0: no preview / not trusted).
struct PreviewTrust {
  int deep;     // for kPreviewAlphaTrusted pixels
  int shallow;  // for kPreviewAlphaShallow pixels
};

#ifdef __CUDACC__
#define FRACTAL_HD __host__ __device__
#else
#define FRACTAL_HD
#endif
FRACTAL_HD inline unsigned short packCount(int taken, int hits) {
  taken = taken > 127 ? 127 : taken;
  hits = hits > 127 ? 127 : hits;
  return (unsigned short)((taken << 8) | hits);
}
FRACTAL_HD inline int samplesTaken(unsigned short m) { return (m >> 8) & 127; }
FRACTAL_HD inline int limitHits(unsigned short m) { return m & 127; }
FRACTAL_HD inline bool isBlack(unsigned short m) { return samplesTaken(m) > 0 && limitHits(m) == samplesTaken(m); }
// Bit 7: every sample of the pixel lies in the main cardioid or period-2 bulb, so it is
// interior for certain and no deeper limit, and no preview, can change that.
FRACTAL_HD inline bool isCertain(unsigned short m) { return (m & 128) != 0; }
// Bit 15: the coarse 4x4-subsampled image hit the limit at the 2x2 coarse points nearest to
// this pixel, so it is taken as interior and not computed (its samples, if any, are kept). It
// is shown as pure black, whatever the zoom preview shows there.
FRACTAL_HD inline bool isCoarseBlack(unsigned short m) { return (m & 0x8000) != 0; }
// Black for display and for the tile census: all samples hit, or skipped by the coarse image.
FRACTAL_HD inline bool showsBlack(unsigned short m) { return isBlack(m) || isCoarseBlack(m); }
FRACTAL_HD inline unsigned short withCertain(unsigned short m, bool certain) {
  return (unsigned short)(certain ? (m | 128) : (m & ~128));
}
// Weight of the zoom preview prior after `passes` passes on the current view: kPreviewWeight
// samples at first, aged like the samples (kSampleAging per pass) and faded out entirely by
// the time every pixel has all its samples, so nothing of the magnified old image lingers.
inline float priorWeight(float weight, int passes, int totalSamples) {
  if (passes >= totalSamples) return 0.0f;
  return weight * powf(kSampleAging, (float)passes) * (1.0f - (float)passes / (float)totalSamples);
}

// Rebuilds the color sums of every pixel from the stored escape values (mu) with the current
// palette, color scale and offset: a recoloring without any iteration. The samples are
// replayed in order with the same aging as the live accumulation. Asynchronous.
void recolorSamples(Accum accum, int w, int h, const ViewParams& p);

// Allocates / frees an accumulation buffer for a w x h image with `samples` samples per
// pixel, orbits in double if `useDouble`. ensureOrbits reallocates the per-sample storage
// when `samples` or the precision changes; the orbits are then forgotten (the caller clears
// the buffer anyway). renderPass does this itself for the precision it uses.
Accum allocAccum(int w, int h, int samples, bool useDouble);
void ensureOrbits(Accum& a, int w, int h, int samples, bool useDouble);
void freeAccum(Accum& a);

// Enables or disables the black heuristics: the deep-interior skip rule (black pixels whose
// nearest 2x2 block of 4x4 tiles is entirely black are not probed again) and the
// black-neighbour rule (a pixel whose four neighbours are black keeps its single sample). On
// by default; JF_NO_TILE_SKIP=1 in the environment and the self-tests turn them off.
void setTileSkip(bool on);

// Resets every pixel to zero samples. Asynchronous.
void clearSamples(Accum accum, int w, int h);

// One pass: every pixel that has fewer than totalSamples(p) samples receives its next sample
// of the pattern; finished pixels are skipped. After totalSamples(p) passes since the last
// clear or shift, every pixel is complete. Synchronous.
// `preview` / `previewIter` (optional): the zoom preview and the iteration limit it was
// computed with. Where the preview was computed at a higher limit than this pass and shows
// color, a pixel whose samples all hit the limit is not taken as black: it is probed again
// every pass (never skipped by the deep-interior rule) and the census does not count it.
// `settledIter`: the limit of the previous pass. Re-probed pixels with a sample whose stored
// orbit is older than that (or has no orbit) were not probed last time: they are newly exposed
// by the eroding black region and counted in exposedCount, so the caller can hold the limit
// until the erosion has caught up.
// `activeSamples`: how many samples of the pattern a pixel may have taken (0: all of them).
// Pixels with fewer take their next sample; pixels at the quota only re-probe their hits. The
// app deepens with one sample per pixel first and adds the others once the depth has settled.
// `earlyAA`: the extra samples are being taken early, at the current (low) limit, for pixels
// whose samples have all escaped so far; a pixel whose new extra sample hits drops back to
// its two primary samples alone and takes no more early samples (it gets them after the
// depth phase). The primary samples are never affected by a dropped extra sample.
// `shallowIter`: the limit the preview's kPreviewAlphaShallow pixels (the surround fill) were
// computed with; they are trusted like the others, but only to that depth.
// `iterBudget`: iterations per sample in this pass at most (0: kIterBudget), see kIterBudget.
// `coarseChase`: the coarse image runs towards the preview's depth (beyond this pass's limit)
// so it can clear trusted preview colors it has verified deeper; false after a zoom in, where
// it tracks the pass's limit only and no computation towards the old depth carries over.
RenderStats renderPass(Accum& accum, int w, int h, const ViewParams& p, const uchar4* preview = nullptr,
                       int previewIter = 0, int settledIter = 0, int activeSamples = 0, bool earlyAA = false,
                       int shallowIter = 0, int iterBudget = 0, bool coarseChase = true);

// Runs the coarse 4x4 image alone, towards `target` with at most `budget` iterations per point
// (the per-sample budget times kCoarseBudgetFactor), resuming as in a pass, and returns its
// statistics. The app probes a new view with this before its first pass, staged from the auto
// limit up to the inherited depth, to predict the first pass's cost from the points' escape
// iterations and the boundary points, and to pick the depth and sample count that fit the
// time budget. The field stays in `accum` for the passes that follow. Synchronous.
CoarseStats coarsePass(Accum& accum, int w, int h, const ViewParams& p, int target, int budget);

// Translation reuse: dst(x, y) = src(x + shiftX, y + shiftY) where that lies inside the
// image, zero samples elsewhere. Asynchronous.
void shiftSamples(Accum src, Accum dst, int w, int h, int shiftX, int shiftY);

// Combines the accumulated samples with the preview into RGBA8 pixels. Where the preview is
// valid (alpha 255) it counts as `previewWeight` extra samples of its color, so the image
// sharpens gradually as real samples arrive:
//   color = (previewWeight * preview + sum of samples) / (previewWeight + sample count).
// Pixels with no samples and no usable preview are black. `preview` may be null. Asynchronous.
// `previewIter` / `sampleIter`: the iteration limits the preview and the latest samples were
// computed with. A pixel whose samples all hit the limit is pure black, unless the preview
// there was computed deeper and shows color, in which case the preview is shown as is. A
// pixel blocked by the coarse image is pure black regardless of the preview. `shallowIter` is the
// depth of the preview's kPreviewAlphaShallow pixels, as for renderPass.
void resolveSamples(Accum accum, const uchar4* preview, float previewWeight, uchar4* out, int w, int h,
                    int previewIter = 0, int sampleIter = 0, int shallowIter = 0);

// Zoom/pan preview: resamples `src`, an image of view `from`, with bicubic (Catmull-Rom)
// interpolation into the pixel grid of view `to`. The alpha channel is a weight: taps are
// weighted by it and the result's alpha is the resampled alpha, so unrendered areas (alpha 0)
// stay weightless; pixels outside `src` become black with alpha 0. Asynchronous.
void resamplePreview(const uchar4* src, uchar4* dst, int w, int h, const ViewParams& from, const ViewParams& to);

// Fills the pixels of `dst` (an image of view `to`, w x h) that have alpha 0 from `src`, an
// image of view `from` with its own size sw x sh (the low-iteration surround image), resampled
// the same way. Filled pixels get alpha kPreviewAlphaShallow at most, so their colors count as
// a prior but are only trusted to the surround's depth. Asynchronous.
void fillPreview(const uchar4* src, int sw, int sh, const ViewParams& from, uchar4* dst, int w, int h,
                 const ViewParams& to);

// Which precision Auto would pick for this view.
bool wantsDouble(const ViewParams& p);
