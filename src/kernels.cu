// CUDA kernels for fractal rendering. Templated on the scalar type so the same
// code produces a fast float path and a precise double path.
#include "fractal.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

namespace {

__device__ __forceinline__ float fractf(float x) { return x - floorf(x); }

__device__ __forceinline__ float3 cosPalette(float t, float3 a, float3 b, float3 c, float3 d) {
  const float k = 6.28318530718f;
  return make_float3(a.x + b.x * __cosf(k * (c.x * t + d.x)),
                     a.y + b.y * __cosf(k * (c.y * t + d.y)),
                     a.z + b.z * __cosf(k * (c.z * t + d.z)));
}

// The well-known "Ultra Fractal" default gradient: deep blue -> white -> orange -> black.
__device__ float3 ultraFractalGradient(float t) {
  const float pos[6] = {0.0f, 0.16f, 0.42f, 0.6425f, 0.8575f, 1.0f};
  const float3 col[6] = {make_float3(0, 7, 100),     make_float3(32, 107, 203), make_float3(237, 255, 255),
                         make_float3(255, 170, 0),   make_float3(0, 2, 0),      make_float3(0, 7, 100)};
  t = fractf(t);
  int k = 0;
  while (k < 4 && t >= pos[k + 1]) ++k;
  float u = (t - pos[k]) / (pos[k + 1] - pos[k]);
  float3 c0 = col[k], c1 = col[k + 1];
  return make_float3((c0.x + (c1.x - c0.x) * u) / 255.0f, (c0.y + (c1.y - c0.y) * u) / 255.0f,
                     (c0.z + (c1.z - c0.z) * u) / 255.0f);
}

__device__ float3 colorize(float mu, int palette, float colorScale, float colorOffset) {
  float t = mu * colorScale + colorOffset;
  const float3 half = make_float3(0.5f, 0.5f, 0.5f);
  switch (palette) {
    default:
    case 0: return ultraFractalGradient(t);
    case 1: return cosPalette(t, half, half, make_float3(1, 1, 1), make_float3(0.00f, 0.10f, 0.20f));
    case 2: return cosPalette(t, half, half, make_float3(1, 1, 1), make_float3(0.00f, 0.33f, 0.67f));
    case 3: return cosPalette(t, half, half, make_float3(1, 1, 0.5f), make_float3(0.80f, 0.90f, 0.30f));
    case 4: return cosPalette(t, half, half, make_float3(2, 1, 0), make_float3(0.50f, 0.20f, 0.25f));
    case 5: {  // monochrome
      float v = 0.5f + 0.5f * __cosf(6.28318530718f * t);
      return make_float3(v, v, v);
    }
  }
}

__device__ __forceinline__ unsigned char toByte(float v) {
  return (unsigned char)(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f + 0.5f);
}

// Points inside the main cardioid or the period-2 bulb never escape; skip them.
template <typename T>
__device__ __forceinline__ bool inMainBulbs(T cr, T ci) {
  T x1 = cr + T(1);
  if (x1 * x1 + ci * ci < T(0.0625)) return true;
  T xq = cr - T(0.25);
  T q = xq * xq + ci * ci;
  return q * (q + xq) < T(0.25) * ci * ci;
}

// The 2-vector type matching the iteration precision, for the stored orbits.
template <typename T> struct Vec2;
template <> struct Vec2<float> { using type = float2; };
template <> struct Vec2<double> { using type = double2; };
__device__ __forceinline__ float2 makeVec2(float x, float y) { return make_float2(x, y); }
__device__ __forceinline__ double2 makeVec2(double x, double y) { return make_double2(x, y); }

// Subpixel offsets of the samples, in the order they are taken (see sampleOrder).
__constant__ float2 c_offsets[kMaxSamples];

// [0]: number of probed pixels whose iteration loop ran all the way to maxIter in the
// current pass; [1]: number of re-probed pixels that had a sample escape (were rescued);
// [2]: the highest iteration count at which a rescued sample escaped; [3]: number of
// re-probed pixels that were newly exposed (a sample lagged behind the previous limit).
__device__ int d_counts[4];

// Adds this warp's statistics to d_counts with one atomic per warp and counter.
__device__ __forceinline__ void countStats(bool hit, bool rescued, int rescueIter, bool exposed) {
  const unsigned mask = __activemask();
  const int nh = __popc(__ballot_sync(mask, hit));
  const int nr = __popc(__ballot_sync(mask, rescued));
  const int mi = __reduce_max_sync(mask, rescued ? rescueIter : 0);
  const int ne = __popc(__ballot_sync(mask, exposed));
  if ((__ffs(mask) - 1) == (int)(threadIdx.x % 32)) {
    if (nh) atomicAdd(&d_counts[0], nh);
    if (nr) atomicAdd(&d_counts[1], nr);
    if (mi) atomicMax(&d_counts[2], mi);
    if (ne) atomicAdd(&d_counts[3], ne);
  }
}

// Population count of black pixels per kTile x kTile tile of the accumulation buffer: pixels
// whose every real sample hit the iteration limit (including ones the skip rule below has
// stopped probing, which keep that state). Only real samples count; a pixel that so far shows
// nothing but the zoom preview has no samples and is never black. Tiles cut off by the image
// edge count only the pixels they have.
constexpr int kTile = 4;
// A pixel's primary samples: taken in the first pass (the second only if the first escaped)
// and carried through the deeper search. Early extra samples beyond them are taken at the
// low limit of the moment and are dropped again, all of them, if one hits that limit.
constexpr int kPrimary = 2;
// Stored in a pixel's third orbit slot while it has only its primary samples: an early extra
// sample hit the limit, so no more early samples for this pixel.
constexpr int kEarlyFailed = -3;

// The depth a preview pixel's colors were tested to, by its alpha class (see fractal.h):
// kPreviewAlphaTrusted carries the previous view's depth, kPreviewAlphaShallow the depth of
// the low-iteration surround image that fills the border of a zoom out, anything else nothing.
__device__ __forceinline__ int previewDepthAt(uchar4 p, PreviewTrust t) {
  return p.w == kPreviewAlphaTrusted ? t.deep : p.w == kPreviewAlphaShallow ? t.shallow : 0;
}

// True when the preview at this pixel was computed with a deeper iteration limit than `limit`
// and shows color there: such a pixel is not black however often it hit, it keeps the
// preview's color and is probed deeper until the limit has passed the preview's.
__device__ __forceinline__ bool previewDeeperAndColored(const uchar4* preview, size_t idx, PreviewTrust t,
                                                        int limit) {
  if (!preview) return false;
  const uchar4 p = preview[idx];
  return previewDepthAt(p, t) > limit && (p.x | p.y | p.z) != 0;
}

// True when the preview's color stands in for a black pixel: computed deeper than the pixel's
// samples reached (above), or trusted (every sample of the previous view's pixel escaped) while
// this pixel has fewer than its full samples: one hit sample cannot refute a full pixel's mean,
// the pixel is a boundary sliver at best, and its antialiasing samples settle it.
__device__ __forceinline__ bool previewHolds(const uchar4* preview, size_t idx, PreviewTrust t, int depth, int taken,
                                             int total) {
  if (!preview) return false;
  const uchar4 p = preview[idx];
  if ((p.x | p.y | p.z) == 0) return false;
  return previewDepthAt(p, t) > depth || (p.w == kPreviewAlphaTrusted && t.deep > 0 && taken < total);
}

__global__ void blackTilesKernel(const unsigned short* __restrict__ meta, const int* __restrict__ depth,
                                 const uchar4* __restrict__ preview, PreviewTrust trust, int total,
                                 unsigned char* __restrict__ tiles, int w, int h, int tw, int th) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= tw || ty >= th) return;
  int black = 0;
  for (int y = ty * kTile; y < min(ty * kTile + kTile, h); ++y)
    for (int x = tx * kTile; x < min(tx * kTile + kTile, w); ++x) {
      const size_t idx = (size_t)y * w + x;
      const unsigned short m = meta[idx];
      black += showsBlack(m) &&
               (isCertain(m) || isCoarseBlack(m) ||
                !previewHolds(preview, idx, trust, depth[idx], samplesTaken(m), total));
    }
  tiles[(size_t)ty * tw + tx] = (unsigned char)black;
}

// True when every pixel of tile (tx, ty) is black. Tiles outside the image count as black,
// so the image edge does not keep a black region alive.
__device__ __forceinline__ bool tileAllBlack(const unsigned char* tiles, int tx, int ty, int w, int h, int tw,
                                             int th) {
  if (tx < 0 || ty < 0 || tx >= tw || ty >= th) return true;
  const int n = min(kTile, w - tx * kTile) * min(kTile, h - ty * kTile);
  return tiles[(size_t)ty * tw + tx] >= n;
}

// True when the 2x2 block of tiles nearest to pixel (x, y) (its own tile and the three
// neighbours on the side the pixel lies on) is entirely black.
__device__ __forceinline__ bool deepInBlack(const unsigned char* tiles, int x, int y, int w, int h, int tw,
                                            int th) {
  const int tx = x / kTile, ty = y / kTile;
  const int nx = (x % kTile) < kTile / 2 ? tx - 1 : tx + 1;
  const int ny = (y % kTile) < kTile / 2 ? ty - 1 : ty + 1;
  return tileAllBlack(tiles, tx, ty, w, h, tw, th) && tileAllBlack(tiles, nx, ty, w, h, tw, th) &&
         tileAllBlack(tiles, tx, ny, w, h, tw, th) && tileAllBlack(tiles, nx, ny, w, h, tw, th);
}

// Per tile: does the (deeper) preview show any color in it? Used to erode the preview's black
// by one tile: a black pixel whose nearest 2x2 block of tiles contains preview color is next
// to the previous view's boundary and keeps being probed, so each zoom refines the boundary.
__global__ void previewTilesKernel(const uchar4* __restrict__ preview, PreviewTrust trust, int limit,
                                   unsigned char* __restrict__ tiles, int w, int h, int tw, int th) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= tw || ty >= th) return;
  int colored = 0;
  for (int y = ty * kTile; y < min(ty * kTile + kTile, h); ++y)
    for (int x = tx * kTile; x < min(tx * kTile + kTile, w); ++x) {
      colored |= previewDeeperAndColored(preview, (size_t)y * w + x, trust, limit);
    }
  tiles[(size_t)ty * tw + tx] = (unsigned char)colored;
}

__device__ __forceinline__ bool tileHasColor(const unsigned char* tiles, int tx, int ty, int tw, int th) {
  return tx >= 0 && ty >= 0 && tx < tw && ty < th && tiles[(size_t)ty * tw + tx] != 0;
}

// True when the deeper preview shows color in the 2x2 block of tiles nearest to pixel (x, y).
__device__ __forceinline__ bool previewColorNearby(const unsigned char* previewTiles, int x, int y, int tw,
                                                   int th) {
  if (!previewTiles) return false;
  const int tx = x / kTile, ty = y / kTile;
  const int nx = (x % kTile) < kTile / 2 ? tx - 1 : tx + 1;
  const int ny = (y % kTile) < kTile / 2 ? ty - 1 : ty + 1;
  return tileHasColor(previewTiles, tx, ty, tw, th) || tileHasColor(previewTiles, nx, ty, tw, th) ||
         tileHasColor(previewTiles, tx, ny, tw, th) || tileHasColor(previewTiles, nx, ny, tw, th);
}

// Gives every unfinished pixel its next sample, accumulated into `accum`.
// Iterates sample `k` of pixel (x, y) up to maxIter, resuming from its stored orbit when
// the slot holds one. Returns the iteration count reached (>= maxIter: hit) and leaves z in
// zr/zi (zr2/zi2 = their squares). Stores the orbit of a hit so the next probe can resume.
template <typename T>
__device__ __forceinline__ int iterateSample(int x, int y, int w, int h, T cx, T cy, T scale, int k, int maxIter,
                                             int budget, typename Vec2<T>::type* orbit, int* orbitIters, size_t slot,
                                             bool& bulb, bool& unfinished, T& zr, T& zi, T& zr2, T& zi2) {
  const float ox = c_offsets[k].x, oy = c_offsets[k].y;
  const T cr = cx + (T(x) + T(ox) - T(w) * T(0.5)) * scale;
  const T ci = cy - (T(y) + T(oy) - T(h) * T(0.5)) * scale;
  zr = zi = zr2 = zi2 = 0;
  unfinished = false;
  bulb = inMainBulbs(cr, ci);
  if (bulb) {
    orbitIters[slot] = INT_MAX;  // known interior at any limit: never counts as lagging
    return maxIter;
  }
  const int done = orbitIters[slot];
  int i = 0;
  if (done > 0) {
    if (done >= maxIter) return done;  // already known to hit this limit
    // Resume this sample's orbit where the last probe left it.
    const typename Vec2<T>::type z = orbit[slot];
    zr = z.x;
    zi = z.y;
    zr2 = zr * zr;
    zi2 = zi * zi;
    i = done;
  }
  // At most `budget` iterations in this pass (kIterBudget, or less when frames run long).
  const int cap = min(maxIter, i + budget);
  // Large bailout radius makes the smooth iteration count accurate.
  const T bailout = T(1 << 16);
  while (i < cap && zr2 + zi2 <= bailout) {
    zi = T(2) * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    ++i;
  }
  if (i >= cap) {
    // Hit the limit, or ran out of budget: keep the orbit for the next probe.
    orbit[slot] = makeVec2(zr, zi);
    orbitIters[slot] = i;
    unfinished = i < maxIter;
  } else if (done > 0) {
    orbitIters[slot] = -1;  // escaped: nothing to resume any more
  }
  return i;
}


// The coarse image: iterates the center point of every 4x4 tile towards `target`, resuming
// its stored orbit, at most `budget` iterations per pass, and records whether it has escaped.
// The target may lie beyond the pixels' limit (the depth a trusted preview's colors were
// tested to), so the coarse image can run ahead of the pixels; iters[] then tells how deep
// each point has been taken, which is what the blocking rule compares against.
template <typename T>
__global__ void coarseKernel(typename Vec2<T>::type* __restrict__ orbit, int* __restrict__ iters,
                             unsigned char* __restrict__ hit, int w, int h, int tw, int th, T cx, T cy, T scale,
                             int target, int budget) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= tw || ty >= th) return;
  const size_t t = (size_t)ty * tw + tx;
  // The tile center, as a pixel position (offset 0 from that pixel's center).
  const T cr = cx + (T(tx * kTile + kTile / 2) - T(w) * T(0.5)) * scale;
  const T ci = cy - (T(ty * kTile + kTile / 2) - T(h) * T(0.5)) * scale;
  if (inMainBulbs(cr, ci)) {
    iters[t] = INT_MAX;
    hit[t] = 1;
    return;
  }
  T zr = 0, zi = 0, zr2 = 0, zi2 = 0;
  int i = 0;
  const int done = iters[t];
  if (done <= kCoarseEscaped) {
    hit[t] = 0;  // escaped earlier: a deeper limit cannot change that
    return;
  }
  if (done > 0) {
    if (done >= target) {
      hit[t] = 1;
      return;
    }
    const typename Vec2<T>::type z = orbit[t];
    zr = z.x;
    zi = z.y;
    zr2 = zr * zr;
    zi2 = zi * zi;
    i = done;
  }
  // The coarse image is small, so its budget is generous: it reaches the pixels' limit in one
  // pass in practice and catches up with a deep preview over a few dozen passes.
  const int cap = min(target, i + budget);
  const T bailout = T(1 << 16);
  while (i < cap && zr2 + zi2 <= bailout) {
    zi = T(2) * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    ++i;
  }
  if (i >= cap) {
    orbit[t] = makeVec2(zr, zi);
    iters[t] = i;  // not escaped so far, taken this deep
    hit[t] = 1;
  } else {
    iters[t] = kCoarseEscaped - i;  // escaped at iteration i: never again (a deeper limit cannot change that)
    hit[t] = 0;
  }
}

// Statistics of the coarse field for planning the first pass of a view (see coarsePass):
// points that have not escaped, boundary points among them (a neighbouring point escaped),
// the iterations the escaped points took in all, and the iterations done by every point.
__device__ unsigned long long d_coarseStats[5];
__global__ void coarseStatsKernel(const int* __restrict__ iters, const unsigned char* __restrict__ hit, int tw,
                                  int th, int target) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= tw || ty >= th) return;
  const size_t t = (size_t)ty * tw + tx;
  const int it = iters[t];
  const bool h = hit[t] != 0;
  unsigned long long hits = h, boundary = 0, escaped = 0, work = 0, lagging = 0;
  if (h) {
    work = it == INT_MAX ? 0 : (unsigned long long)max(it, 0);
    lagging = it != INT_MAX && it < target;  // not escaped, not yet at the target
    for (int dy = -1; dy <= 1 && !boundary; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = tx + dx, ny = ty + dy;
        if (nx < 0 || ny < 0 || nx >= tw || ny >= th) continue;
        if (!hit[(size_t)ny * tw + nx]) {
          boundary = 1;
          break;
        }
      }
  } else if (it <= kCoarseEscaped) {
    escaped = work = (unsigned long long)(kCoarseEscaped - it);
  }
  // One atomic per warp and counter.
  for (int k = 0; k < 5; ++k) {
    const unsigned long long v = k == 0 ? hits : k == 1 ? boundary : k == 2 ? escaped : k == 3 ? work : lagging;
    unsigned long long sum = v;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0 && sum) atomicAdd(&d_coarseStats[k], sum);
  }
}

// True when the coarse point of tile (tx, ty) has not escaped within `depth` iterations.
__device__ __forceinline__ bool coarseHitAt(const unsigned char* hit, const int* iters, int tx, int ty, int tw,
                                            int th, int depth) {
  if (tx < 0 || ty < 0 || tx >= tw || ty >= th) return true;  // beyond the image: as if interior
  const size_t t = (size_t)ty * tw + tx;
  return hit[t] != 0 && iters[t] >= depth;
}

// True when the 2x2 coarse points nearest to pixel (x, y) have all been taken at least `depth`
// iterations deep without escaping. The blocker thus only affects pixels whose own depth (the
// limit they probe to, or the depth a trusted preview's colors were tested to) does not
// exceed the depth the coarse image was computed to.
__device__ __forceinline__ bool coarseInterior(const unsigned char* hit, const int* iters, int x, int y, int tw,
                                               int th, int depth) {
  const int tx = x / kTile, ty = y / kTile;
  const int nx = (x % kTile) < kTile / 2 ? tx - 1 : tx + 1;
  const int ny = (y % kTile) < kTile / 2 ? ty - 1 : ty + 1;
  return coarseHitAt(hit, iters, tx, ty, tw, th, depth) && coarseHitAt(hit, iters, nx, ty, tw, th, depth) &&
         coarseHitAt(hit, iters, tx, ny, tw, th, depth) && coarseHitAt(hit, iters, nx, ny, tw, th, depth);
}

// Adds one sample (color c at weight wgt) to the accumulation, aged by kSampleAging (1: a
// plain sum). Written with explicit fused multiply-adds so that every path (live and replay)
// rounds identically.
__device__ __forceinline__ void accumulateSample(float4& a, float3 c, float wgt) {
  a.x = __fmaf_rn(a.x, kSampleAging, c.x);
  a.y = __fmaf_rn(a.y, kSampleAging, c.y);
  a.z = __fmaf_rn(a.z, kSampleAging, c.z);
  a.w = __fmaf_rn(a.w, kSampleAging, wgt);
}

// Whether a hit sample's black counts: only once it has been taken as deep as the pixel is
// resolved to (`depth`); a sample still short of that is pending and adds nothing.
__device__ __forceinline__ bool hitCounts(int reached, int depth) { return reached >= depth; }

// A pixel's color sum and weight from its samples, replayed in order with the same sequence
// as the live accumulation: hits that count add black at kLimitHitWeight, pending hits
// nothing, escapes their color at weight 1. Used after every re-probe and for recoloring, so
// all paths stay identical.
__device__ __forceinline__ float4 replayPixel(int taken, unsigned mask, const float* __restrict__ muBuf,
                                              const int* __restrict__ orbitIters, size_t slotBase, int depth,
                                              int palette, float colorScale, float colorOffset) {
  float4 a = make_float4(0, 0, 0, 0);
  for (int k = 0; k < taken; ++k) {
    if ((mask >> k) & 1u) {
      if (hitCounts(orbitIters[slotBase + (size_t)k], depth))
        accumulateSample(a, make_float3(0, 0, 0), kLimitHitWeight);
    } else {
      accumulateSample(a, colorize(muBuf[slotBase + (size_t)k], palette, colorScale, colorOffset), 1.0f);
    }
  }
  return a;
}

// Gives every unfinished pixel its next sample, accumulated into `accum`; pixels with all
// samples taken re-probe their limit-hit samples at the current limit.
template <typename T>
__global__ void __launch_bounds__(256)
mandelbrotKernel(float4* __restrict__ accum, unsigned short* __restrict__ meta,
                 unsigned short* __restrict__ hitMask, int* __restrict__ depth,
                 typename Vec2<T>::type* __restrict__ orbit,
                 int* __restrict__ orbitIters, float* __restrict__ muBuf, int orbitSlots,
                 const unsigned char* __restrict__ coarseHit, const int* __restrict__ coarseIters,
                 const unsigned char* __restrict__ tiles, const unsigned char* __restrict__ previewTiles,
                 const uchar4* __restrict__ preview, PreviewTrust trust, int w, int h, int tw, int th, T cx, T cy,
                 T scale, int totalSamples,
                 int activeSamples, bool earlyAA, int maxIter, int budget, int settledIter, int palette,
                 float colorScale, float colorOffset) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const size_t idx = (size_t)y * w + x;
  unsigned short m = meta[idx];
  const int taken = samplesTaken(m), hits = limitHits(m);
  unsigned mask = hitMask[idx];
  // The coarse image says interior all around this pixel: take it as black without computing
  // anything. Only for pixels with no escaped sample, though: a known color is never
  // overridden (a thin filament between the coarse points would otherwise vanish and
  // flicker). A pixel showing a trusted deeper preview's color claims that depth, so only
  // coarse points taken at least that deep can block it; blocked, it is cleared to black
  // whatever interpolated color the earlier rendering left there (see resolveKernel).
  if (coarseHit && (taken == 0 || isBlack(m))) {
    // The pixel's claim: the preview's depth where it shows a trusted color deeper than the
    // pixel's own samples reached, else the pass's limit.
    const int claim = previewHolds(preview, idx, trust, depth[idx], taken, totalSamples)
                          ? max(previewDepthAt(preview[idx], trust), maxIter)
                          : maxIter;
    if (coarseInterior(coarseHit, coarseIters, x, y, tw, th, claim)) {
      if (!isCoarseBlack(m)) meta[idx] = (unsigned short)(m | 0x8000);
      return;
    }
  }
  if (isCoarseBlack(m)) {
    m = (unsigned short)(m & ~0x8000);  // no longer covered by the coarse image: compute normally
    meta[idx] = m;
  }
  // With its quota of samples taken, a pixel is only revisited to re-probe the samples that
  // hit the limit, now at a deeper limit.
  bool reprobe = taken >= activeSamples;
  const size_t slotBase = idx * (size_t)orbitSlots;
  // Early extra samples go only to pixels whose samples have all escaped and that have not
  // failed an early sample before; the others just re-probe their hits meanwhile.
  if (!reprobe && earlyAA && taken >= kPrimary &&
      (mask != 0 || (orbitSlots > kPrimary && orbitIters[slotBase + kPrimary] == kEarlyFailed)))
    reprobe = true;
  if (reprobe && mask == 0) return;
  // A pixel whose four neighbours are black keeps its single sample: more subpixels there
  // would almost surely run to the limit too, at full cost. (Neighbours outside the image
  // count as black.) Off together with the tile heuristic (tiles == nullptr).
  if (!reprobe && taken >= 1 && tiles) {
    const bool l = x == 0 || isBlack(meta[idx - 1]), r = x == w - 1 || isBlack(meta[idx + 1]);
    const bool u = y == 0 || isBlack(meta[idx - w]), d = y == h - 1 || isBlack(meta[idx + w]);
    if (l && r && u && d) return;
  }
  // A black pixel (every sample so far hit the limit) with nothing but black around it is not
  // worth probing deeper: it keeps its state, so it stays black for the tile census and the
  // black region is only eroded from its edge. Not within a tile of the deeper preview's
  // color, though: that band along the previous view's boundary is probed deeper every time.
  if (isBlack(m) && tiles &&
      (isCertain(m) ||
       (!previewHolds(preview, idx, trust, depth[idx], taken, totalSamples) &&
        !previewColorNearby(previewTiles, x, y, tw, th))) &&
      deepInBlack(tiles, x, y, w, h, tw, th))
    return;
  bool anyHit = false, exposed = false;
  bool certain = isCertain(m);
  if (!reprobe) {
    // New samples, in sequence up to the quota, stopping at the first that hits the limit:
    // the first pass takes two samples unless the first one hits.
    float4 a = accum[idx];
    int t = taken, nh = hits;
    int dep = depth[idx];  // how deep the pixel's samples so far are resolved
    while (t < activeSamples) {
      T zr, zi, zr2, zi2;
      bool bulb, unfinished;
      const int i = iterateSample<T>(x, y, w, h, cx, cy, scale, t, maxIter, budget, orbit, orbitIters,
                                     slotBase + (size_t)t, bulb, unfinished, zr, zi, zr2, zi2);
      const bool hit = i >= maxIter || unfinished;
      if (hit && earlyAA && t >= kPrimary) {
        // An early extra sample hit the (low) limit: drop the extra samples and keep the
        // primary ones alone; the rest come after the depth phase at the settled depth.
        orbitIters[slotBase + (size_t)t] = -1;  // the orbit this probe stored
        orbitIters[slotBase + kPrimary] = kEarlyFailed;
        accum[idx] = replayPixel(kPrimary, 0u, muBuf, orbitIters, slotBase, dep, palette, colorScale, colorOffset);
        meta[idx] = packCount(kPrimary, 0);
        hitMask[idx] = 0;
        depth[idx] = dep;
        return;
      }
      certain = (t == 0 || certain) && bulb;  // all samples so far known interior
      exposed |= unfinished;
      if (hit) {
        // Reached the limit (or the budget): black, but only if taken as deep as the pixel's
        // earlier samples were resolved; short of that it is pending and adds nothing yet.
        anyHit |= !bulb;
        mask |= 1u << t;
        if (hitCounts(i, dep)) accumulateSample(a, make_float3(0, 0, 0), kLimitHitWeight);
        dep = max(dep, i);
        ++t;
        ++nh;
        break;
      }
      // Continuous escape time: mu = i + 1 - log2(log2|z|) = i + 2 - log2(log2|z|^2).
      const float mu = (float)i + 2.0f - log2f(log2f((float)(zr2 + zi2)));
      muBuf[slotBase + (size_t)t] = mu;
      accumulateSample(a, colorize(mu, palette, colorScale, colorOffset), 1.0f);
      dep = max(dep, maxIter);  // escaped: resolved to this pass's limit (and beyond)
      ++t;
    }
    countStats(anyHit, false, 0, exposed);
    accum[idx] = a;
    meta[idx] = withCertain(packCount(t, nh), certain);
    hitMask[idx] = (unsigned short)mask;
    depth[idx] = dep;
    return;
  }

  // Every hit (or pending) sample gets another go, resumed from its stored orbit.
  int escaped = 0, maxEscape = 0;
  int dep = depth[idx];
  for (unsigned rest = mask; rest;) {
    const int k = __ffs(rest) - 1;
    rest &= ~(1u << k);
    const size_t slot = slotBase + (size_t)k;
    // Not probed in the previous pass: this pixel was just exposed by the eroding black.
    exposed |= orbitIters[slot] < settledIter;
    T zr, zi, zr2, zi2;
    bool bulb, unfinished;
    const int i = iterateSample<T>(x, y, w, h, cx, cy, scale, k, maxIter, budget, orbit, orbitIters, slot, bulb,
                                   unfinished,
                                   zr, zi, zr2, zi2);
    exposed |= unfinished;
    if (i >= maxIter || unfinished) {
      anyHit |= !bulb;
      dep = max(dep, i);
    } else {
      muBuf[slot] = (float)i + 2.0f - log2f(log2f((float)(zr2 + zi2)));
      ++escaped;
      maxEscape = max(maxEscape, i);
      mask &= ~(1u << k);
      dep = max(dep, maxIter);
    }
  }
  countStats(anyHit, escaped > 0, maxEscape, exposed);
  // Rescued samples replace black ones and pending samples may have caught up: replay the
  // pixel so the sums stay canonical.
  depth[idx] = dep;
  accum[idx] = replayPixel(taken, mask, muBuf, orbitIters, slotBase, dep, palette, colorScale, colorOffset);
  if (escaped == 0) return;  // still interior at this limit: nothing else new
  meta[idx] = packCount(taken, hits - escaped);  // escaped: not certain
  hitMask[idx] = (unsigned short)mask;
}

// Replays a pixel's samples in order with the current colors: the same age-then-add
// sequence as the live accumulation, so a recolored buffer continues seamlessly.
__global__ void recolorKernel(float4* __restrict__ accum, const unsigned short* __restrict__ meta,
                              const unsigned short* __restrict__ hitMask, const float* __restrict__ muBuf,
                              const int* __restrict__ orbitIters, const int* __restrict__ depth, int orbitSlots,
                              int n, int palette, float colorScale, float colorOffset) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  accum[i] = replayPixel(samplesTaken(meta[i]), hitMask[i], muBuf, orbitIters, (size_t)i * orbitSlots, depth[i],
                         palette, colorScale, colorOffset);
}

__global__ void resolveKernel(const float4* __restrict__ accum, const unsigned short* __restrict__ meta,
                              const int* __restrict__ depth, const uchar4* __restrict__ preview,
                              float previewWeight, PreviewTrust trust, int total, uchar4* __restrict__ out, int w,
                              int h) {
  const int n = w * h;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 a = accum[i];
  const unsigned short m = meta[i];
  // A pixel whose samples all hit the limit (the black heuristic's notion of black, kept by
  // pixels the skip rule stopped probing) is shown as pure black, unless the preview of the
  // same area was computed deeper and shows color: then that color is shown as is. That
  // includes a pixel whose samples so far lie in the main cardioid or bulb (certain): on the
  // boundary its other samples may well escape, and the preview's antialiased color is the
  // better estimate until they are in. A pixel blocked by the coarse 4x4 image is pure black
  // whatever the preview says: no interpolated halo of an earlier rendering survives there.
  if (showsBlack(m)) {
    // Deeper than what this pixel's own samples have reached (not the pass's limit: with a
    // deep start the samples catch up over several budgeted passes).
    if (!isCoarseBlack(m) && previewHolds(preview, i, trust, depth[i], samplesTaken(m), total)) {
      uchar4 p = preview[i];
      p.w = 255;
      out[i] = p;
    } else {
      out[i] = make_uchar4(0, 0, 0, 255);
    }
    return;
  }
  const float cnt = a.w;  // total weight of the aged samples
  // The pixel's own alpha class once it has samples: trusted when every sample escaped and no
  // neighbour is black, mixed when some sample hit or a neighbour is black: such a boundary
  // pixel may cover a sliver of black its samples missed, and its color must not be held
  // against hits after a zoom any more than a part-black pixel's.
  bool nextToBlack = false;
  if (limitHits(m) == 0) {
    const int x = i % w, y = i / w;
    for (int dy = -1; dy <= 1 && !nextToBlack; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        if (showsBlack(meta[(size_t)ny * w + nx])) {
          nextToBlack = true;
          break;
        }
      }
  }
  const unsigned char own = limitHits(m) > 0 || nextToBlack ? kPreviewAlphaMixed : kPreviewAlphaTrusted;
  uchar4 p = preview ? preview[i] : make_uchar4(0, 0, 0, 0);
  // The output alpha is the pixel's confidence: 255 once it has real samples, otherwise the
  // preview's own alpha. A pixel nobody has rendered stays at alpha 0, so that when it is
  // resampled into a later view it carries no weight at all.
  if (p.w == 0) {
    // No usable preview here: plain average of the real samples (black at count 0).
    float inv = cnt > 0 ? 1.0f / cnt : 0.0f;
    out[i] = make_uchar4(toByte(a.x * inv), toByte(a.y * inv), toByte(a.z * inv), cnt > 0 ? own : 0);
    return;
  }
  // The preview stands in for previewWeight samples, scaled by its alpha; until real samples
  // arrive it is shown as is.
  const float wp = cnt > 0 ? previewWeight * (p.w / 255.0f) : 1.0f;
  const float inv = 1.0f / (wp + cnt);
  const float s = wp / 255.0f;
  out[i] = make_uchar4(toByte((a.x + s * p.x) * inv), toByte((a.y + s * p.y) * inv), toByte((a.z + s * p.z) * inv),
                       cnt > 0 ? own : p.w);
}

// Catmull-Rom cubic weight for a tap at distance t from the sample point.
__device__ __forceinline__ float cubicWeight(float t) {
  t = fabsf(t);
  if (t < 1.0f) return (1.5f * t - 2.5f) * t * t + 1.0f;
  if (t < 2.0f) return ((-0.5f * t + 2.5f) * t - 4.0f) * t + 2.0f;
  return 0.0f;
}

// dst(x, y) = bicubic sample of src at (r*x + bx, r*y + by), the position of this pixel's
// center in the source image.
// Resamples `src` (sw x sh) into `dst` (w x h): dst pixel (x, y) reads src at (r x + bx, r y + by).
// Black and non-black areas are kept separate: the nearest source pixel decides whether the
// result is black (a rendered pixel with no color), and the color of a non-black result is
// interpolated from the non-black taps only. So a magnified black region keeps its boundary
// instead of being fringed with smeared boundary colors (which the trusted-preview rule would
// then hold as a halo until the deepening caught up), and the boundary colors are not darkened
// by black taps either. With `fillOnly`, only dst pixels of alpha 0 (nothing there yet) are
// written, and the result's alpha is capped at kPreviewAlphaShallow: the fill is a prior, but
// its colors are only trusted to the shallow depth (see fillPreview).
__global__ void resampleKernel(const uchar4* __restrict__ src, int sw, int sh, uchar4* __restrict__ dst, int w,
                               int h, float r, float bx, float by, bool fillOnly) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  if (fillOnly && dst[(size_t)y * w + x].w != 0) return;
  const float px = fmaf(r, (float)x, bx), py = fmaf(r, (float)y, by);
  uchar4 result = make_uchar4(0, 0, 0, 0);  // alpha 0: outside the source image
  if (px >= -0.5f && px <= sw - 0.5f && py >= -0.5f && py <= sh - 0.5f) {
    const int ix = (int)floorf(px), iy = (int)floorf(py);
    const float fx = px - ix, fy = py - iy;
    float wx[4] = {cubicWeight(fx + 1), cubicWeight(fx), cubicWeight(1 - fx), cubicWeight(2 - fx)};
    float wy[4] = {cubicWeight(fy + 1), cubicWeight(fy), cubicWeight(1 - fy), cubicWeight(2 - fy)};
    // The class of the nearest source pixel: black (rendered, no color) or not.
    const int nx = min(max((int)floorf(px + 0.5f), 0), sw - 1), ny = min(max((int)floorf(py + 0.5f), 0), sh - 1);
    const uchar4 nearest = src[(size_t)ny * sw + nx];
    const bool blackHere = nearest.w != 0 && (nearest.x | nearest.y | nearest.z) == 0;
    // Alpha-weighted taps: source pixels with alpha 0 were never rendered and must not pull
    // the result toward black. The output alpha is the resampled alpha, so unrendered areas
    // stay at weight 0 through any number of successive zooms. Taps of the other class are
    // left out of the color (their weight still counts towards the alpha).
    float3 sum = make_float3(0, 0, 0);
    float wa = 0, wc = 0;
    for (int j = 0; j < 4; ++j) {
      const int sy = min(max(iy - 1 + j, 0), sh - 1);
      for (int i = 0; i < 4; ++i) {
        const int sx = min(max(ix - 1 + i, 0), sw - 1);
        const uchar4 c = src[(size_t)sy * sw + sx];
        const float wgt = wx[i] * wy[j] * c.w;
        wa += wgt;
        const bool black = c.w != 0 && (c.x | c.y | c.z) == 0;
        if (black != blackHere) continue;
        sum.x += wgt * c.x;
        sum.y += wgt * c.y;
        sum.z += wgt * c.z;
        wc += wgt;
      }
    }
    if (wa > 0.0f) {
      const float inv = wc > 0.0f ? 1.0f / (wc * 255.0f) : 0.0f;
      // The alpha is the resampled weight where the area is only partly rendered; where it is
      // fully covered it is the nearest source pixel's class (trusted, shallow or mixed), so
      // the classes do not blur into each other (averaging 255 and 253 would read as 254).
      const unsigned char alpha = wa >= 0.99f * 255.0f && nearest.w != 0 ? nearest.w : toByte(wa / 255.0f);
      result = make_uchar4(toByte(sum.x * inv), toByte(sum.y * inv), toByte(sum.z * inv), alpha);
      if (blackHere) result.x = result.y = result.z = 0;
      if (fillOnly) result.w = min(result.w, (unsigned char)kPreviewAlphaShallow);
    }
  }
  dst[(size_t)y * w + x] = result;
}

__global__ void shiftKernel(Accum src, Accum dst, int w, int h, int sx, int sy) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  const int ox = x + sx, oy = y + sy;
  const size_t i = (size_t)y * w + x;
  const bool inside = ox >= 0 && ox < w && oy >= 0 && oy < h;
  const size_t o = inside ? (size_t)oy * w + ox : 0;
  dst.color[i] = inside ? src.color[o] : make_float4(0, 0, 0, 0);
  dst.meta[i] = inside ? src.meta[o] : (unsigned short)0;
  dst.hitMask[i] = inside ? src.hitMask[o] : (unsigned short)0;
  dst.depth[i] = inside ? src.depth[o] : 0;
  // Both buffers have the same slot count and precision (the app keeps them in step).
  const int slots = dst.orbitSlots;
  const int units = dst.orbitDouble ? 2 : 1;  // 8-byte units per orbit
  unsigned long long* dOrbit = (unsigned long long*)dst.orbit;
  const unsigned long long* sOrbit = (const unsigned long long*)src.orbit;
  for (int s = 0; s < slots; ++s) {
    dst.orbitIters[i * slots + s] = inside ? src.orbitIters[o * slots + s] : -1;
    dst.mu[i * slots + s] = inside ? src.mu[o * slots + s] : 0.0f;
    for (int u = 0; u < units; ++u)
      dOrbit[(i * slots + s) * units + u] = inside ? sOrbit[(o * slots + s) * units + u] : 0ull;
  }
}

int clampSamples(int n) {
  n = std::min(std::max(n, 1), kMaxSamples);
  int p = 1;
  while (p < n) p <<= 1;  // up to a power of two: 1, 2, 4, 8, 16
  return p;
}

// Fills `out` with the aa*aa subpixel offsets (in [-0.5, 0.5) about the pixel center) in the
// order samples are taken. The pattern is a greedy largest-gap sequence: the first sample is
// the pixel center, and every further sample goes to the point of the pixel square farthest
// from all earlier ones (toroidal distance, so the pattern tiles seamlessly across
// neighbouring pixels), nudged by a random amount of up to kJitter of that gap radius so the
// result is a little chaotic rather than a grid. After any number of samples the ones taken
// so far are spread evenly over the pixel. The random seed is fixed, so renders repeat.
void sampleOrder(int n, std::vector<float2>& out) {
  constexpr int kGrid = 256;      // candidate positions per axis
  constexpr float kJitter = 0.3f;  // fraction of the gap radius a sample may move
  std::vector<float> gap((size_t)kGrid * kGrid, 1e9f);  // squared toroidal distance to the nearest sample
  std::mt19937 rng(1234u + (unsigned)n);
  std::uniform_real_distribution<float> uni(0.0f, 1.0f);
  out.clear();
  auto add = [&](float x, float y) {
    out.push_back(make_float2(x - 0.5f, y - 0.5f));
    for (int gy = 0; gy < kGrid; ++gy) {
      float dy = std::fabs((gy + 0.5f) / kGrid - y);
      dy = std::min(dy, 1.0f - dy);
      for (int gx = 0; gx < kGrid; ++gx) {
        float dx = std::fabs((gx + 0.5f) / kGrid - x);
        dx = std::min(dx, 1.0f - dx);
        float& g = gap[(size_t)gy * kGrid + gx];
        g = std::min(g, dx * dx + dy * dy);
      }
    }
  };
  add(0.5f, 0.5f);
  while ((int)out.size() < n) {
    size_t best = 0;
    for (size_t i = 1; i < gap.size(); ++i)
      if (gap[i] > gap[best]) best = i;
    const float r = std::sqrt(gap[best]);
    const float angle = 6.2831853f * uni(rng), dist = kJitter * r * std::sqrt(uni(rng));
    float x = (best % kGrid + 0.5f) / kGrid + dist * std::cos(angle);
    float y = (best / kGrid + 0.5f) / kGrid + dist * std::sin(angle);
    x -= std::floor(x);
    y -= std::floor(y);
    add(x, y);
  }
}

// Makes sure the constant-memory offset table matches `aa`.
void uploadOffsets(int n) {
  static int current = 0;
  if (current == n) return;
  std::vector<float2> order;
  sampleOrder(n, order);
  cudaMemcpyToSymbol(c_offsets, order.data(), order.size() * sizeof(float2));
  current = n;
}

dim3 gridFor(int w, int h, dim3 block) { return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y); }

template <typename T>
void launch(Accum accum, const unsigned char* tiles, const unsigned char* previewTiles, const uchar4* preview,
            PreviewTrust trust, int w, int h, int tw, int th, const ViewParams& p, int total, int activeSamples,
            bool earlyAA, int settledIter, int budget, bool coarseChase) {
  dim3 block(32, 8);
  // The coarse image first (cheap: one point per tile, resumed), then the pixels. With
  // coarseChase it runs towards the deeper of the pass's limit and the trusted preview's
  // depth, so that it can clear preview colors that were tested deeper than the pixels have
  // been probed so far; without it (after a zoom in) it tracks the pass's limit only, so no
  // computation towards the previous view's depth carries over into the new view.
  const unsigned char* coarseHit = nullptr;
  static const bool noCoarse = getenv("JF_NO_COARSE") != nullptr;  // A/B switch for the coarse image
  if (tiles && !noCoarse) {
    const int target = coarseChase ? max(p.maxIter, max(trust.deep, trust.shallow)) : p.maxIter;
    coarseKernel<T><<<gridFor(accum.tw, accum.th, block), block>>>(
        (typename Vec2<T>::type*)accum.coarseOrbit, accum.coarseIters, accum.coarseHit, w, h, accum.tw, accum.th,
        (T)p.centerX, (T)p.centerY, (T)p.scale, target, kCoarseBudgetFactor * budget);
    coarseHit = accum.coarseHit;
  }
  mandelbrotKernel<T><<<gridFor(w, h, block), block>>>(accum.color, accum.meta, accum.hitMask, accum.depth,
                                                      (typename Vec2<T>::type*)accum.orbit, accum.orbitIters,
                                                      accum.mu, accum.orbitSlots, coarseHit, accum.coarseIters,
                                                      tiles, previewTiles, preview, trust, w, h, tw, th,
                                                      (T)p.centerX,
                                                      (T)p.centerY, (T)p.scale, total, activeSamples, earlyAA,
                                                      p.maxIter, budget, settledIter, p.palette, p.colorScale,
                                                      p.colorOffset);
}

bool g_tileSkip = true;

// Device buffers for the per-tile black counts and preview-color flags, grown as needed.
// Returns false (and null tiles) when the skip rule is off or memory is short.
bool tileMaps(const unsigned short* meta, const int* depth, const uchar4* preview, PreviewTrust trust, int limit,
              int total, int w, int h, int& tw, int& th, unsigned char*& blackOut, unsigned char*& previewOut) {
  static unsigned char* tiles = nullptr;
  static unsigned char* previewTiles = nullptr;
  static size_t capacity = 0;
  blackOut = previewOut = nullptr;
  if (!g_tileSkip) return false;
  tw = (w + kTile - 1) / kTile;
  th = (h + kTile - 1) / kTile;
  const size_t need = (size_t)tw * th;
  if (need > capacity) {
    if (tiles) cudaFree(tiles);
    if (previewTiles) cudaFree(previewTiles);
    tiles = previewTiles = nullptr;
    capacity = 0;
    if (cudaMalloc(&tiles, need) != cudaSuccess || cudaMalloc(&previewTiles, need) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    capacity = need;
  }
  dim3 block(32, 8);
  blackTilesKernel<<<gridFor(tw, th, block), block>>>(meta, depth, preview, trust, total, tiles, w, h, tw, th);
  blackOut = tiles;
  if (preview && max(trust.deep, trust.shallow) > limit) {
    previewTilesKernel<<<gridFor(tw, th, block), block>>>(preview, trust, limit, previewTiles, w, h, tw, th);
    previewOut = previewTiles;
  }
  return true;
}

}  // namespace

int totalSamples(const ViewParams& p) {
  return clampSamples(p.samples);
}

bool wantsDouble(const ViewParams& p) {
  if (p.precision == Precision::Double) return true;
  if (p.precision == Precision::Float) return false;
  return p.scale < kDoubleThreshold;
}

void setTileSkip(bool on) { g_tileSkip = on; }

void samplePattern(int n, std::vector<float2>& out) { sampleOrder(clampSamples(n), out); }

// Per-sample storage: one slot per sample of the pattern, orbits in the view's precision.
void allocOrbits(Accum& a, int w, int h, int samples, bool useDouble) {
  const size_t n = (size_t)w * h * samples;
  if (cudaMalloc(&a.orbit, n * (useDouble ? sizeof(double2) : sizeof(float2))) != cudaSuccess ||
      cudaMalloc(&a.orbitIters, n * sizeof(int)) != cudaSuccess || cudaMalloc(&a.mu, n * sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "CUDA error: cannot allocate per-sample storage for a %dx%d image with %d samples; use fewer "
            "samples (--aa) or a smaller window\n", w, h, samples);
    exit(1);
  }
  cudaMemset(a.orbitIters, 0xFF, n * sizeof(int));  // -1: nothing stored
  a.orbitSlots = samples;
  a.orbitDouble = useDouble;
  a.tw = (w + kTile - 1) / kTile;
  a.th = (h + kTile - 1) / kTile;
  const size_t nt = (size_t)a.tw * a.th;
  if (cudaMalloc(&a.coarseOrbit, nt * (useDouble ? sizeof(double2) : sizeof(float2))) != cudaSuccess ||
      cudaMalloc(&a.coarseIters, nt * sizeof(int)) != cudaSuccess ||
      cudaMalloc(&a.coarseHit, nt) != cudaSuccess) {
    fprintf(stderr, "CUDA error: cannot allocate the coarse image\n");
    exit(1);
  }
  cudaMemset(a.coarseIters, 0xFF, nt * sizeof(int));
  cudaMemset(a.coarseHit, 0, nt);
}

void freeOrbits(Accum& a) {
  if (a.orbit) cudaFree(a.orbit);
  if (a.orbitIters) cudaFree(a.orbitIters);
  if (a.mu) cudaFree(a.mu);
  if (a.coarseOrbit) cudaFree(a.coarseOrbit);
  if (a.coarseIters) cudaFree(a.coarseIters);
  if (a.coarseHit) cudaFree(a.coarseHit);
  a.orbit = a.coarseOrbit = nullptr;
  a.orbitIters = a.coarseIters = nullptr;
  a.mu = nullptr;
  a.coarseHit = nullptr;
  a.orbitSlots = 0;
}

Accum allocAccum(int w, int h, int samples, bool useDouble) {
  Accum a;
  if (cudaMalloc(&a.color, (size_t)w * h * sizeof(float4)) != cudaSuccess ||
      cudaMalloc(&a.meta, (size_t)w * h * sizeof(unsigned short)) != cudaSuccess ||
      cudaMalloc(&a.hitMask, (size_t)w * h * sizeof(unsigned short)) != cudaSuccess ||
      cudaMalloc(&a.depth, (size_t)w * h * sizeof(int)) != cudaSuccess) {
    fprintf(stderr, "CUDA error: cannot allocate a %dx%d accumulation buffer\n", w, h);
    exit(1);
  }
  cudaMemset(a.depth, 0, (size_t)w * h * sizeof(int));
  allocOrbits(a, w, h, samples, useDouble);
  return a;
}

void ensureOrbits(Accum& a, int w, int h, int samples, bool useDouble) {
  if (a.orbitSlots == samples && a.orbitDouble == useDouble) return;
  freeOrbits(a);
  allocOrbits(a, w, h, samples, useDouble);
}

void freeAccum(Accum& a) {
  if (a.color) cudaFree(a.color);
  if (a.meta) cudaFree(a.meta);
  if (a.hitMask) cudaFree(a.hitMask);
  if (a.depth) cudaFree(a.depth);
  freeOrbits(a);
  a = Accum();
}

void clearSamples(Accum accum, int w, int h) {
  cudaMemsetAsync(accum.color, 0, (size_t)w * h * sizeof(float4));
  cudaMemsetAsync(accum.meta, 0, (size_t)w * h * sizeof(unsigned short));
  cudaMemsetAsync(accum.hitMask, 0, (size_t)w * h * sizeof(unsigned short));
  cudaMemsetAsync(accum.depth, 0, (size_t)w * h * sizeof(int));
  cudaMemsetAsync(accum.orbitIters, 0xFF, (size_t)w * h * accum.orbitSlots * sizeof(int));  // -1: nothing stored
  cudaMemsetAsync(accum.coarseIters, 0xFF, (size_t)accum.tw * accum.th * sizeof(int));
  cudaMemsetAsync(accum.coarseHit, 0, (size_t)accum.tw * accum.th);
}

RenderStats renderPass(Accum& accum, int w, int h, const ViewParams& p, const uchar4* preview, int previewIter,
                       int settledIter, int activeSamples, bool earlyAA, int shallowIter, int iterBudget,
                       bool coarseChase) {
  const PreviewTrust trust = {preview ? previewIter : 0, preview ? shallowIter : 0};
  if (iterBudget <= 0) iterBudget = kIterBudget;
  const int total = totalSamples(p);
  if (activeSamples <= 0 || activeSamples > total) activeSamples = total;
  ensureOrbits(accum, w, h, total, wantsDouble(p));  // the per-sample storage must match the pass
  static cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (!e0) {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
  }
  uploadOffsets(total);

  RenderStats st;
  st.usedDouble = wantsDouble(p);
  int zero[4] = {0, 0, 0, 0}, counts[4] = {0, 0, 0, 0};
  cudaMemcpyToSymbolAsync(d_counts, zero, sizeof zero);
  cudaEventRecord(e0);
  int tw = 0, th = 0;
  unsigned char *tiles = nullptr, *previewTiles = nullptr;
  tileMaps(accum.meta, accum.depth, preview, trust, p.maxIter, total, w, h, tw, th, tiles, previewTiles);
  if (st.usedDouble)
    launch<double>(accum, tiles, previewTiles, preview, trust, w, h, tw, th, p, total, activeSamples, earlyAA,
                   settledIter, iterBudget, coarseChase);
  else
    launch<float>(accum, tiles, previewTiles, preview, trust, w, h, tw, th, p, total, activeSamples, earlyAA,
                  settledIter, iterBudget, coarseChase);
  cudaEventRecord(e1);
  cudaError_t err = cudaMemcpyFromSymbol(counts, d_counts, sizeof counts);  // also waits for the pass
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) fprintf(stderr, "CUDA error in renderPass: %s\n", cudaGetErrorString(err));
  st.hitLimit = counts[0] != 0;
  st.hitCount = counts[0];
  st.rescueCount = counts[1];
  st.maxRescueIter = counts[2];
  st.exposedCount = counts[3];
  cudaEventElapsedTime(&st.kernelMs, e0, e1);
  return st;
}

template <typename T>
void launchCoarse(Accum& accum, int w, int h, const ViewParams& p, int target, int budget) {
  dim3 block(32, 8);
  coarseKernel<T><<<gridFor(accum.tw, accum.th, block), block>>>(
      (typename Vec2<T>::type*)accum.coarseOrbit, accum.coarseIters, accum.coarseHit, w, h, accum.tw, accum.th,
      (T)p.centerX, (T)p.centerY, (T)p.scale, target, budget);
}

CoarseStats coarsePass(Accum& accum, int w, int h, const ViewParams& p, int target, int budget) {
  CoarseStats st;
  if (budget <= 0) budget = kCoarseBudgetFactor * kIterBudget;
  ensureOrbits(accum, w, h, totalSamples(p), wantsDouble(p));
  static cudaEvent_t e0 = nullptr, e1 = nullptr;
  if (!e0) {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
  }
  unsigned long long zero[5] = {0, 0, 0, 0, 0}, sums[5] = {0, 0, 0, 0, 0};
  cudaMemcpyToSymbolAsync(d_coarseStats, zero, sizeof zero);
  cudaEventRecord(e0);
  if (wantsDouble(p))
    launchCoarse<double>(accum, w, h, p, target, budget);
  else
    launchCoarse<float>(accum, w, h, p, target, budget);
  cudaEventRecord(e1);
  dim3 block(32, 8);
  coarseStatsKernel<<<gridFor(accum.tw, accum.th, block), block>>>(accum.coarseIters, accum.coarseHit, accum.tw,
                                                                   accum.th, target);
  cudaError_t err = cudaMemcpyFromSymbol(sums, d_coarseStats, sizeof sums);
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) fprintf(stderr, "CUDA error in coarsePass: %s\n", cudaGetErrorString(err));
  cudaEventElapsedTime(&st.kernelMs, e0, e1);
  st.points = accum.tw * accum.th;
  st.hits = (int)sums[0];
  st.boundary = (int)sums[1];
  st.escapedIters = (double)sums[2];
  st.work = (double)sums[3];
  st.complete = sums[4] == 0;  // no point short of the target that has not escaped
  return st;
}

__global__ void coarseShiftKernel(Accum src, Accum dst, int sx, int sy) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= dst.tw || ty >= dst.th) return;
  const int ox = tx + sx, oy = ty + sy;
  const bool inside = ox >= 0 && ox < src.tw && oy >= 0 && oy < src.th;
  const size_t t = (size_t)ty * dst.tw + tx, o = inside ? (size_t)oy * src.tw + ox : 0;
  dst.coarseIters[t] = inside ? src.coarseIters[o] : -1;
  dst.coarseHit[t] = inside ? src.coarseHit[o] : 0;
  const int units = dst.orbitDouble ? 2 : 1;
  for (int u = 0; u < units; ++u)
    ((unsigned long long*)dst.coarseOrbit)[t * units + u] =
        inside ? ((const unsigned long long*)src.coarseOrbit)[o * units + u] : 0ull;
}

void shiftSamples(Accum src, Accum dst, int w, int h, int shiftX, int shiftY) {
  dim3 block(32, 8);
  shiftKernel<<<gridFor(w, h, block), block>>>(src, dst, w, h, shiftX, shiftY);
  if (shiftX % kTile == 0 && shiftY % kTile == 0) {
    coarseShiftKernel<<<gridFor(dst.tw, dst.th, block), block>>>(src, dst, shiftX / kTile, shiftY / kTile);
  } else {
    // The coarse points no longer line up with the tiles: start the coarse image over.
    cudaMemsetAsync(dst.coarseIters, 0xFF, (size_t)dst.tw * dst.th * sizeof(int));
    cudaMemsetAsync(dst.coarseHit, 0, (size_t)dst.tw * dst.th);
  }
}

void recolorSamples(Accum accum, int w, int h, const ViewParams& p) {
  const int n = w * h;
  recolorKernel<<<(n + 255) / 256, 256>>>(accum.color, accum.meta, accum.hitMask, accum.mu, accum.orbitIters,
                                          accum.depth, accum.orbitSlots, n, p.palette, p.colorScale, p.colorOffset);
}

void resolveSamples(Accum accum, const uchar4* preview, float previewWeight, uchar4* out, int w, int h,
                    int previewIter, int sampleIter, int shallowIter) {
  int n = w * h;
  const PreviewTrust trust = {preview ? previewIter : 0, preview ? shallowIter : 0};
  (void)sampleIter;  // the comparison is per pixel, against the depth its samples reached
  resolveKernel<<<(n + 255) / 256, 256>>>(accum.color, accum.meta, accum.depth, preview, previewWeight, trust,
                                          accum.orbitSlots, out, w, h);
}

static void resample(const uchar4* src, int sw, int sh, const ViewParams& from, uchar4* dst, int w, int h,
                     const ViewParams& to, bool fillOnly) {
  // Pixel x of `to` sits at world to.centerX + (x - w/2) * to.scale; the same point in `from`
  // is at pixel (world - from.centerX) / from.scale + sw/2. Likewise for y with the sign flipped.
  const double r = to.scale / from.scale;
  const double bx = (to.centerX - from.centerX) / from.scale + 0.5 * sw - 0.5 * w * r;
  const double by = (from.centerY - to.centerY) / from.scale + 0.5 * sh - 0.5 * h * r;
  dim3 block(32, 8);
  resampleKernel<<<gridFor(w, h, block), block>>>(src, sw, sh, dst, w, h, (float)r, (float)bx, (float)by,
                                                  fillOnly);
}

void resamplePreview(const uchar4* src, uchar4* dst, int w, int h, const ViewParams& from, const ViewParams& to) {
  resample(src, w, h, from, dst, w, h, to, false);
}

void fillPreview(const uchar4* src, int sw, int sh, const ViewParams& from, uchar4* dst, int w, int h,
                 const ViewParams& to) {
  resample(src, sw, sh, from, dst, w, h, to, true);
}
