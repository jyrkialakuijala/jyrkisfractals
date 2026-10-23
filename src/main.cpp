// JyrkisFractals: interactive CUDA fractal viewer.
// SDL2 window + OpenGL presentation, CUDA renders straight into a GL pixel buffer.
// Antialiasing is progressive: one sample per pixel while the view moves, the remaining
// samples of a largest-gap pattern accumulate once the view is still.
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "fractal.h"
#include "image_io.h"

namespace {

// GL 1.5 buffer-object entry points, loaded at runtime.
PFNGLGENBUFFERSPROC pglGenBuffers;
PFNGLBINDBUFFERPROC pglBindBuffer;
PFNGLBUFFERDATAPROC pglBufferData;
PFNGLDELETEBUFFERSPROC pglDeleteBuffers;

bool loadGL() {
  pglGenBuffers = (PFNGLGENBUFFERSPROC)SDL_GL_GetProcAddress("glGenBuffers");
  pglBindBuffer = (PFNGLBINDBUFFERPROC)SDL_GL_GetProcAddress("glBindBuffer");
  pglBufferData = (PFNGLBUFFERDATAPROC)SDL_GL_GetProcAddress("glBufferData");
  pglDeleteBuffers = (PFNGLDELETEBUFFERSPROC)SDL_GL_GetProcAddress("glDeleteBuffers");
  return pglGenBuffers && pglBindBuffer && pglBufferData && pglDeleteBuffers;
}

#define CUDA_CHECK(call)                                                                        \
  do {                                                                                          \
    cudaError_t err_ = (call);                                                                  \
    if (err_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

const char* kHelp =
    "Controls:\n"
    "  mouse wheel        zoom in/out around the cursor\n"
    "  left drag          pan\n"
    "  double click       zoom in 2x at cursor;  right click: zoom out 2x\n"
    "  Z / X              zoom in / out around the center\n"
    "  PgUp / PgDn        zoom in / out around the mouse pointer\n"
    "  arrow keys         pan\n"
    "  1 .. 9, 0          antialiasing: samples per pixel (1 = off, 0 = 16; default 7)\n"
    "  + / -              more / fewer iterations (disables auto)\n"
    "  A                  toggle automatic iteration count\n"
    "  D                  cycle precision: auto / float / double\n"
    "  P / Shift+P        next / previous palette\n"
    "  [ / ]              compress / stretch the palette\n"
    "  , / .              rotate the palette\n"
    "  S                  save screenshot (PNG, fully antialiased)\n"
    "  F                  toggle fullscreen\n"
    "  R                  reset view\n"
    "  H                  print this help;  Esc / Q: quit\n";

// Kernel time we are willing to spend per displayed frame while accumulating samples.
const double kFrameBudgetMs = 8.0;
// If a full pass costs more than this, a resampled preview of the previous image is shown
// first and the real samples follow; below it the first pass is quick enough to wait for.
const double kPreviewThresholdMs = 3.0;
// After a zoom or pan the resampled previous image counts as this many samples per pixel, so
// the real samples sharpen it gradually instead of replacing it after the first pass.
const float kPreviewWeight = 8.0f;
// Progressive deepening: whenever a pass has pixels that ran to the iteration limit, the
// effective limit for the next pass grows by this much. The adjustment resets on each zoom.
const int kIterStep = 500;
// Passes (and the deepening) continue past the last antialiasing sample as long as at least
// this many pixels still ran to the limit in the latest pass.
const int kFewHits = 5;
// Colors of a previous view keep standing in for black pixels (see previewIter) as long as
// they have not been magnified by more than this factor in total; beyond it the black areas
// are computed from scratch.
const double kPreviewTrust = 5.0;
// The effective iteration limit never exceeds this.
const int kMaxIter = 1 << 22;
// The surround image (see Renderer::renderSurround): a half-resolution image of a field this
// many times wider than the view, at least this many iterations (more where the auto limit of
// that field is higher, but never more than one pass's budget), rendered whenever the view
// leaves the field it covers. It fills in the border a zoom out or a pan uncovers.
const int kSurroundField = 4;
const int kSurroundIter = 1000;
// --test-flicker caps the deepening here: deep enough for the whole schedule, quick to reach.
const int kFlickerTestCap = 1 << 15;
// An update (passes, resolve and display of one frame) that took longer than this in real time
// lowers the per-pass iteration budget of the next passes (see Renderer::adaptBudget), aiming
// at kFrameTargetMs; updates faster than half the target raise it back, up to kIterBudget.
const double kSlowFrameMs = 100.0;
const double kFrameTargetMs = 50.0;
const int kMinIterBudget = 64;
// Before the first pass of a zoomed view the coarse 4x4 field probes it (Renderer::planFirstPass):
// the probing may spend this much kernel time, and the first pass is planned (depth, samples)
// to stay within this predicted time.
const double kProbeBudgetMs = 10.0;
const double kFirstPassTargetMs = 30.0;
// A zoom in may start up to this many times deeper than the depth the previous image needed
// (never deeper than it verified): boundary pixels that need a little more depth then catch up
// through the per-pass budget in a few passes instead of 500 iterations per pass, so they
// are black for fewer frames, while the truly black ones are not chased towards the cap.
const int kDepthHeadroom = 4;

struct Options {
  int width = 1280, height = 800;
  std::string headlessOut;  // if set: render once without a window, write PNG, exit
  int benchFrames = 0;      // if > 0: headless timing loop
  int exitAfterMs = 0;      // GUI smoke test: quit automatically
  std::string saveOnExit;   // GUI smoke test: write the converged view to this PNG before quitting
  bool dumpSamples = false; // print the sample pattern for --aa and exit
  int autoZoomSteps = 0;    // GUI test: zoom 1.25x at the center this many times (negative: out), one every 400 ms
  bool deeperTest = false;  // headless: verify that a deeper preview keeps limit-hit pixels colored
  bool resumeTest = false;  // headless: verify that resumed orbits match a from-scratch render
  bool recolorTest = false; // headless: verify that recoloring kept samples matches a fresh render
  bool shiftTest = false;   // headless: verify translation reuse
  int shiftDx = 0, shiftDy = 0;
  bool zoomTest = false;    // headless: verify the zoom preview
  bool flickerTest = false; // headless: verify that no black appears temporarily while a zoom renders
  bool haloTest = false;    // headless: verify that quick zooms leave no temporary color inside black areas
  int haloSteps = 4;
  double zoomFactor = 2.0;
  ViewParams view;
  bool viewGiven = false;
  bool iterGiven = false;
};

void usage() {
  printf(
      "jyrkisfractals [options]\n"
      "  --size WxH          window / image size (default 1280x800)\n"
      "  --center X Y        center of view in the complex plane\n"
      "  --scale S           complex units per pixel\n"
      "  --zoom Z            magnification relative to the default view (alternative to --scale)\n"
      "  --iter N            max iterations (disables auto)\n"
      "  --aa N              antialiasing, samples per pixel, 1..%d, rounded up to a power of two (default 8)\n"
      "  --palette N         palette 0..%d\n"
      "  --precision MODE    auto | float | double\n"
      "  --headless out.png  render one frame to a PNG without opening a window\n"
      "  --bench N           headless: render N full frames and report timing\n"
      "  --exit-after MS     quit the GUI automatically after MS milliseconds\n"
      "  --test-shift DX DY  headless: verify pan reuse against a direct render\n"
      "  --test-zoom F DX DY headless: verify the zoom preview (factor F, offset DX DY source pixels)\n"
      "  --test-flicker F   headless: zoom by F (F < 1: out) as the GUI does and verify that no pixel the\n"
      "                     preview showed colored turns black in between\n"
      "  --test-halo F N    headless: zoom in by F N times quickly (one pass between zooms) and verify that\n"
      "                     no pixel that is black at the end showed color in between (halos in black areas)\n"
      "  --save-on-exit F.png with --exit-after: write the converged view before quitting\n"
      "  --test-deeper      headless: verify that a preview computed deeper keeps limit-hit pixels colored\n"
      "  --test-resume      headless: verify that deepening with resumed orbits matches a fresh render\n"
      "  --test-recolor     headless: verify that recoloring kept samples matches a fresh render\n"
      "  --auto-zoom N      GUI test: zoom 1.25x at the center N times (N < 0: zoom out), one every 400 ms\n"
      "                     (JF_ZOOM_MS in the environment changes the interval)\n"
      "  --dump-samples     print the subpixel sample pattern for --aa and exit\n\n%s",
      kMaxSamples, kPaletteCount - 1, kHelp);
}

double defaultScale(int w) { return 3.5 / w; }

Options parseArgs(int argc, char** argv) {
  Options o;
  double zoom = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](int n) {
      if (i + n >= argc) {
        fprintf(stderr, "missing argument for %s\n", a.c_str());
        exit(2);
      }
    };
    if (a == "--size") {
      need(1);
      if (sscanf(argv[++i], "%dx%d", &o.width, &o.height) != 2 || o.width < 1 || o.height < 1) {
        fprintf(stderr, "bad --size\n");
        exit(2);
      }
    } else if (a == "--center") {
      need(2);
      o.view.centerX = atof(argv[++i]);
      o.view.centerY = atof(argv[++i]);
      o.viewGiven = true;
    } else if (a == "--scale") {
      need(1);
      o.view.scale = atof(argv[++i]);
      o.viewGiven = true;
    } else if (a == "--zoom") {
      need(1);
      zoom = atof(argv[++i]);
      o.viewGiven = true;
    } else if (a == "--iter") {
      need(1);
      o.view.maxIter = atoi(argv[++i]);
      o.iterGiven = true;
    } else if (a == "--aa") {
      need(1);
      o.view.samples = std::min(std::max(atoi(argv[++i]), 1), kMaxSamples);
    } else if (a == "--palette") {
      need(1);
      o.view.palette = atoi(argv[++i]);
    } else if (a == "--precision") {
      need(1);
      std::string m = argv[++i];
      o.view.precision = m == "float" ? Precision::Float : m == "double" ? Precision::Double : Precision::Auto;
    } else if (a == "--headless") {
      need(1);
      o.headlessOut = argv[++i];
    } else if (a == "--bench") {
      need(1);
      o.benchFrames = atoi(argv[++i]);
    } else if (a == "--exit-after") {
      need(1);
      o.exitAfterMs = atoi(argv[++i]);
    } else if (a == "--test-deeper") {
      o.deeperTest = true;
    } else if (a == "--test-resume") {
      o.resumeTest = true;
    } else if (a == "--test-recolor") {
      o.recolorTest = true;
    } else if (a == "--auto-zoom") {
      need(1);
      o.autoZoomSteps = atoi(argv[++i]);
    } else if (a == "--dump-samples") {
      o.dumpSamples = true;
    } else if (a == "--save-on-exit") {
      need(1);
      o.saveOnExit = argv[++i];
    } else if (a == "--test-shift") {
      need(2);
      o.shiftTest = true;
      o.shiftDx = atoi(argv[++i]);
      o.shiftDy = atoi(argv[++i]);
    } else if (a == "--test-flicker") {
      need(1);
      o.flickerTest = true;
      o.zoomFactor = atof(argv[++i]);
    } else if (a == "--test-halo") {
      need(2);
      o.haloTest = true;
      o.zoomFactor = atof(argv[++i]);
      o.haloSteps = atoi(argv[++i]);
    } else if (a == "--test-zoom") {
      need(3);
      o.zoomTest = true;
      o.zoomFactor = atof(argv[++i]);
      o.shiftDx = atoi(argv[++i]);
      o.shiftDy = atoi(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      usage();
      exit(0);
    } else {
      fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      exit(2);
    }
  }
  if (zoom > 0) o.view.scale = defaultScale(o.width) / zoom;
  return o;
}

int autoIterations(double zoom) {
  double z = std::max(zoom, 1.0);
  double it = 256.0 * std::pow(1.0 + 0.5 * std::log10(z), 1.5);
  return (int)std::min(std::max(it, 64.0), (double)(1 << 20));
}

std::string timestampName(const char* ext) {
  char buf[64];
  time_t t = time(nullptr);
  strftime(buf, sizeof buf, "mandelbrot_%Y%m%d_%H%M%S", localtime(&t));
  return std::string(buf) + ext;
}

// Runs passes until every pixel has all its samples; returns total kernel time and precision.
// One pass per sample, then as many passes as it takes for every budgeted sample (see
// kIterBudget) to reach the limit or escape.
RenderStats finishSamples(Accum accum, int w, int h, const ViewParams& v, int passesDone = 0) {
  RenderStats total;
  RenderStats st;
  for (int s = passesDone; s < totalSamples(v); ++s) {
    st = renderPass(accum, w, h, v);
    total.kernelMs += st.kernelMs;
    total.usedDouble = st.usedDouble;
  }
  while (st.exposedCount > 0) {
    st = renderPass(accum, w, h, v);
    total.kernelMs += st.kernelMs;
  }
  return total;
}

// Renders the view from scratch with all samples.
RenderStats renderAllSamples(Accum accum, int w, int h, const ViewParams& v) {
  clearSamples(accum, w, h);
  return finishSamples(accum, w, h, v);
}

// Per-pixel total sample weights and packed counters (see samplesTaken, limitHits,
// isCertain) from an accumulation buffer.
void readCounts(Accum accum, int w, int h, std::vector<float>& weights, std::vector<unsigned short>& meta) {
  std::vector<float4> a((size_t)w * h);
  meta.resize(a.size());
  CUDA_CHECK(cudaMemcpy(a.data(), accum.color, a.size() * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(meta.data(), accum.meta, meta.size() * sizeof(unsigned short), cudaMemcpyDeviceToHost));
  weights.resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) weights[i] = a[i].w;
}

// Self-test for recoloring: render with one palette, recolor the kept samples to another, and
// compare with a fresh render in that palette. The replay is the same sequence of operations
// as the live accumulation, so the images must match exactly.
int testRecolor(Accum accum, uchar4* dev, int w, int h, ViewParams v) {
  const size_t bytes = (size_t)w * h * 4;
  std::vector<uint8_t> recolored(bytes), fresh(bytes);
  renderAllSamples(accum, w, h, v);
  ViewParams other = v;
  other.palette = (v.palette + 1) % kPaletteCount;
  other.colorScale *= 1.5f;
  other.colorOffset += 0.1f;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  recolorSamples(accum, w, h, other);  // once untimed: the first launch of a kernel includes its loading
  cudaEventRecord(e0);
  recolorSamples(accum, w, h, other);
  cudaEventRecord(e1);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(recolored.data(), dev, bytes, cudaMemcpyDeviceToHost));
  float recolorMs = 0;
  cudaEventElapsedTime(&recolorMs, e0, e1);
  const float freshMs = renderAllSamples(accum, w, h, other).kernelMs;
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(fresh.data(), dev, bytes, cudaMemcpyDeviceToHost));
  size_t wrong = 0;
  int maxDiff = 0;
  for (size_t i = 0; i < bytes; i += 4)
    for (int c = 0; c < 3; ++c) {
      const int d = std::abs((int)recolored[i + c] - (int)fresh[i + c]);
      wrong += d != 0;
      maxDiff = std::max(maxDiff, d);
    }
  const bool ok = maxDiff <= 1;
  printf("recolor test (palette %d -> %d): %zu bytes differ (max %d); recolor %.3f ms, fresh render %.3f ms -> %s\n",
         v.palette, other.palette, wrong, maxDiff, recolorMs, freshMs, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// Self-test for orbit resumption: render at a low limit, then deepen in steps of kIterStep
// with re-probing (which resumes stored orbits) up to a high limit, and compare with a fresh
// render at the high limit. The set of black pixels must be identical, since a resumed
// orbit is bit-exact, and the colors must agree closely (the aged weights differ slightly).
int testResume(Accum accum, uchar4* dev, int w, int h, ViewParams v) {
  setTileSkip(false);  // also disables the black-neighbour rule, which makes sample counts neighbour dependent
  const size_t bytes = (size_t)w * h * 4;
  const int low = std::max(v.maxIter / 8, 50), high = v.maxIter;
  v.maxIter = low;
  renderAllSamples(accum, w, h, v);
  float stepMs = 0;
  int steps = 0;
  for (int limit = low + kIterStep; limit <= high; limit += kIterStep, ++steps) {
    v.maxIter = limit;
    stepMs += renderPass(accum, w, h, v).kernelMs;
  }
  v.maxIter = high;
  // Enough passes at the final limit that every hit sample of every pixel has been re-probed.
  for (int s = 0; s < totalSamples(v); ++s) renderPass(accum, w, h, v);
  std::vector<uint8_t> resumed(bytes), fresh(bytes);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(resumed.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> weights;
  std::vector<unsigned short> meta, metaFresh;
  readCounts(accum, w, h, weights, meta);
  const float freshMs = renderAllSamples(accum, w, h, v).kernelMs;
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(fresh.data(), dev, bytes, cudaMemcpyDeviceToHost));
  readCounts(accum, w, h, weights, metaFresh);

  size_t hitMismatch = 0, black = 0;
  double diff = 0;
  for (size_t i = 0; i < meta.size(); ++i) {
    hitMismatch += limitHits(meta[i]) != limitHits(metaFresh[i]);
    black += isBlack(metaFresh[i]);
    for (int c = 0; c < 3; ++c) diff += std::abs((int)resumed[i * 4 + c] - (int)fresh[i * 4 + c]);
  }
  diff /= 3.0 * meta.size();
  const bool ok = hitMismatch == 0 && diff < 2.0;
  printf("resume test (%d -> %d iterations in %d steps): %zu pixels differ in hit count (%zu black), mean color "
         "difference %.3f; deepening took %.2f ms, a fresh render at %d takes %.2f ms -> %s\n",
         low, high, steps, hitMismatch, black, diff, stepMs, high, freshMs, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// Self-test for the deeper-preview rule: render view A at a high limit, zoom in to view B with
// the preview of A, and render B at a much lower limit. Pixels colored in the preview must not
// come out black (they keep the preview color and are probed instead), while without the
// preview's limit they would.
int testDeeper(Accum accum, uchar4* dev, uchar4* dev2, int w, int h, ViewParams a) {
  const size_t bytes = (size_t)w * h * 4;
  a.maxIter = std::max(a.maxIter, 2000);
  ViewParams b = a;
  b.scale = a.scale / 1.5;
  b.maxIter = a.maxIter / 4;
  renderAllSamples(accum, w, h, a);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  resamplePreview(dev, dev2, w, h, a, b);
  std::vector<uint8_t> preview(bytes), with(bytes), without(bytes);
  CUDA_CHECK(cudaMemcpy(preview.data(), dev2, bytes, cudaMemcpyDeviceToHost));

  // The hit counts of a pass with all samples taken show how many black pixels are still
  // being probed: with the rule, the pixels under preview color plus the one-tile band
  // around it; without it, only the edge of the black region.
  clearSamples(accum, w, h);
  for (int s = 0; s < totalSamples(b); ++s) renderPass(accum, w, h, b, dev2, a.maxIter);
  const int probedWith = renderPass(accum, w, h, b, dev2, a.maxIter).hitCount;
  resolveSamples(accum, dev2, kPreviewWeight, dev, w, h, a.maxIter, b.maxIter);
  CUDA_CHECK(cudaMemcpy(with.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> weights;
  std::vector<unsigned short> meta;
  readCounts(accum, w, h, weights, meta);  // the state behind the image with the rule
  clearSamples(accum, w, h);
  for (int s = 0; s < totalSamples(b); ++s) renderPass(accum, w, h, b);
  const int probedWithout = renderPass(accum, w, h, b).hitCount;
  resolveSamples(accum, dev2, kPreviewWeight, dev, w, h, 0, b.maxIter);
  CUDA_CHECK(cudaMemcpy(without.data(), dev, bytes, cudaMemcpyDeviceToHost));

  size_t colored = 0, blackWith = 0, blackWithout = 0, kept = 0, certain = 0, certainWrong = 0, coarse = 0,
         coarseWrong = 0;
  for (size_t i = 0; i < bytes; i += 4) {
    // Only the preview's trusted colors (every sample escaped) are held; mixed boundary pixels
    // (some samples hit) are a prior only.
    if (preview[i + 3] != kPreviewAlphaTrusted || (preview[i] | preview[i + 1] | preview[i + 2]) == 0) continue;
    ++colored;
    const bool bw = (with[i] | with[i + 1] | with[i + 2]) == 0;
    const bool bo = (without[i] | without[i + 1] | without[i + 2]) == 0;
    if (isCertain(meta[i / 4])) {
      // Inside the main cardioid / period-2 bulb so far: the deeper preview's antialiased
      // color stands in all the same (its other samples may escape), unless the coarse image
      // blocked the pixel.
      ++certain;
      certainWrong += isCoarseBlack(meta[i / 4]) ? !bw : memcmp(&with[i], &preview[i], 3) != 0;
      continue;
    }
    if (isCoarseBlack(meta[i / 4])) {
      // Blocked by the coarse 4x4 image: cleared to black, whatever the preview shows.
      ++coarse;
      coarseWrong += !bw;
      continue;
    }
    blackWith += bw;
    blackWithout += bo;
    // A pixel that went black without the rule shows exactly the preview color with it when
    // its samples all hit; it may also have been skipped by the coarse image without the rule
    // and, computed with it, turn out to have escaping samples.
    if (bo) kept += memcmp(&with[i], &preview[i], 3) == 0;
  }
  const bool ok = blackWith == 0 && blackWithout > 0 && kept > 0 && certainWrong == 0 && coarseWrong == 0;
  printf("deeper test (A %d iterations -> B %d): %zu preview-colored pixels, black without the rule %zu, "
         "black with it %zu, showing the preview color %zu; certain interior under preview color %zu (%zu not "
         "showing it); blocked by the coarse image under preview color %zu (%zu not black); black pixels still "
         "probed: %d with the rule, %d without -> %s\n",
         a.maxIter, b.maxIter, colored, blackWithout, blackWith, kept, certain, certainWrong, coarse, coarseWrong,
         probedWith, probedWithout, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// Self-test for translation reuse: render, shift by (dx, dy) pixels, complete the exposed
// strip, and compare against a direct render of the shifted view. Pixels that were kept
// must match the original render exactly; the strip can differ from the direct render only
// by the rounding noise that also shows up in the kept pixels (the pan changes the center
// by a rounded amount, which perturbs chaotic pixels slightly either way).
int testShift(Accum accum, Accum accum2, uchar4* dev, int w, int h, ViewParams v, int dx, int dy) {
  setTileSkip(false);  // the skip rule is history dependent; this tests the reuse alone
  const size_t bytes = (size_t)w * h * 4;
  std::vector<uint8_t> orig(bytes), reuse(bytes), direct(bytes);
  renderAllSamples(accum, w, h, v);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(orig.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> weights;
  std::vector<unsigned short> meta;
  readCounts(accum, w, h, weights, meta);
  shiftSamples(accum, accum2, w, h, dx, dy);
  v.centerX += dx * v.scale;
  v.centerY -= dy * v.scale;
  RenderStats st = finishSamples(accum2, w, h, v);
  resolveSamples(accum2, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(reuse.data(), dev, bytes, cudaMemcpyDeviceToHost));
  renderAllSamples(accum, w, h, v);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(direct.data(), dev, bytes, cudaMemcpyDeviceToHost));

  size_t keptMismatch = 0, kept = 0, strip = 0;
  double keptNoise = 0, stripNoise = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int sx = x + dx, sy = y + dy;
      const size_t i = ((size_t)y * w + x) * 4;
      const bool inside = sx >= 0 && sx < w && sy >= 0 && sy < h;
      // Pixels with limit hits get those samples re-probed after the shift, at coordinates that
      // round differently, so a chaotic sample may flip; only pixels without hits must match.
      const bool settled = inside && limitHits(meta[(size_t)sy * w + sx]) == 0;
      for (int c = 0; c < 3; ++c) {
        if (inside) {
          if (settled) keptMismatch += reuse[i + c] != orig[((size_t)sy * w + sx) * 4 + c];
          keptNoise += std::abs((int)reuse[i + c] - (int)direct[i + c]);
        } else {
          stripNoise += std::abs((int)reuse[i + c] - (int)direct[i + c]);
        }
      }
      (inside ? kept : strip)++;
    }
  }
  keptNoise /= std::max<size_t>(kept * 3, 1);
  stripNoise /= std::max<size_t>(strip * 3, 1);
  const bool ok = keptMismatch == 0 && stripNoise <= std::max(1.0, 4.0 * keptNoise);
  printf("shift test (%d, %d): %zu kept pixels (%zu bytes wrong), %zu new pixels computed in %.3f ms; "
         "mean |reuse - direct| kept %.4f, new %.4f -> %s\n",
         dx, dy, kept, keptMismatch, strip, st.kernelMs, keptNoise, stripNoise, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

// Self-test for the zoom preview: render view A, resample it into view B (zoomed by `factor`
// about the center and moved by (dx, dy) source pixels), and compare with a direct render of
// B over the pixels the preview could cover. With factor 1 the resampling must be exact.
// If `out` is set, the preview and the direct render are written side by side.
int testZoom(Accum accum, uchar4* dev, uchar4* dev2, int w, int h, ViewParams a, double factor, int dx, int dy,
             const std::string& out) {
  const size_t bytes = (size_t)w * h * 4;
  std::vector<uint8_t> preview(bytes), direct(bytes);
  ViewParams b = a;
  b.centerX += dx * a.scale;
  b.centerY -= dy * a.scale;
  b.scale = a.scale / factor;
  renderAllSamples(accum, w, h, a);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  resamplePreview(dev, dev2, w, h, a, b);
  CUDA_CHECK(cudaMemcpy(preview.data(), dev2, bytes, cudaMemcpyDeviceToHost));
  renderAllSamples(accum, w, h, b);
  resolveSamples(accum, nullptr, 0.0f, dev, w, h);
  CUDA_CHECK(cudaMemcpy(direct.data(), dev, bytes, cudaMemcpyDeviceToHost));

  double sum = 0;
  size_t covered = 0, exact = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Position of this pixel's center in the source image.
      double sx = (x - 0.5 * w) / factor + 0.5 * w + dx, sy = (y - 0.5 * h) / factor + 0.5 * h + dy;
      if (sx < 1.5 || sx > w - 2.5 || sy < 1.5 || sy > h - 2.5) continue;  // skip borders
      ++covered;
      const size_t i = ((size_t)y * w + x) * 4;
      int d = 0;
      for (int c = 0; c < 3; ++c) d += std::abs((int)preview[i + c] - (int)direct[i + c]);
      sum += d / 3.0;
      exact += d == 0;
    }
  }
  const double mean = sum / std::max<size_t>(covered, 1);

  // The blended result must be (W * preview + n * direct) / (W + n), with n the pixel's own
  // effective sample weight (limit hits count kLimitHitWeight each).
  std::vector<uint8_t> blended(bytes);
  const float prior = priorWeight(kPreviewWeight, 4, totalSamples(b));  // as after 4 passes
  resolveSamples(accum, dev2, prior, dev, w, h);
  CUDA_CHECK(cudaMemcpy(blended.data(), dev, bytes, cudaMemcpyDeviceToHost));
  std::vector<float> weights;
  std::vector<unsigned short> meta;
  readCounts(accum, w, h, weights, meta);
  int blendErr = 0;
  for (size_t i = 0; i < bytes; i += 4) {
    // The preview weight scales with its alpha.
    const double wp = prior * preview[i + 3] / 255.0;
    const double n = weights[i / 4];
    for (int c = 0; c < 3; ++c) {
      // Pixels whose samples all hit the limit are pure black whatever the preview shows.
      double expect = showsBlack(meta[i / 4]) ? 0.0 : (wp * preview[i + c] + n * (double)direct[i + c]) / (wp + n);
      blendErr = std::max(blendErr, (int)std::abs(blended[i + c] - std::lround(expect)));
    }
  }
  // Zooming out leaves a border nobody has rendered. Resolved without samples it must come out
  // at alpha 0, and a further zoom out must resample it as alpha 0 again (weight 0), never as
  // an opaque black prior.
  size_t borderBad = 0, border = 0, chainBad = 0, chain = 0;
  if (factor < 1.0) {
    std::vector<uint8_t> shown(bytes), chained(bytes);
    clearSamples(accum, w, h);
    resolveSamples(accum, dev2, kPreviewWeight, dev, w, h);  // what the screen shows right after the zoom
    CUDA_CHECK(cudaMemcpy(shown.data(), dev, bytes, cudaMemcpyDeviceToHost));
    ViewParams c = b;
    c.scale = b.scale / factor;
    resamplePreview(dev, dev2, w, h, b, c);  // zoom out again from the shown image
    CUDA_CHECK(cudaMemcpy(chained.data(), dev2, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < bytes; i += 4) {
      if (preview[i + 3] == 0) {
        ++border;
        borderBad += shown[i + 3] != 0 || shown[i] || shown[i + 1] || shown[i + 2];
      }
      const size_t px = (i / 4) % w, py = (i / 4) / w;
      // Position in b of this pixel of c; outside b's covered area (plus a tap margin) the
      // chained preview must carry no weight.
      const double sx = (px - 0.5 * w) / factor + 0.5 * w, sy = (py - 0.5 * h) / factor + 0.5 * h;
      const double cx = (sx - 0.5 * w) / factor + 0.5 * w + dx, cy = (sy - 0.5 * h) / factor + 0.5 * h + dy;
      const bool outsideA = cx < -2.5 || cx > w + 1.5 || cy < -2.5 || cy > h + 1.5;
      if (outsideA) {
        ++chain;
        chainBad += chained[i + 3] != 0;
      }
    }
  }
  // With factor 1 the copy is exact except for chaotic pixels perturbed by the rounded center.
  const bool ok = (factor == 1.0 ? exact >= covered * 99 / 100 : mean < 12.0) && blendErr <= 1 && borderBad == 0 &&
                  chainBad == 0;
  printf("zoom test (x%g, offset %d,%d): %zu covered pixels, %zu exact, mean |preview - direct| %.3f, "
         "blend error %d, unrendered border %zu px (%zu not weightless), chained zoom-out %zu px (%zu not "
         "weightless) -> %s\n",
         factor, dx, dy, covered, exact, mean, blendErr, border, borderBad, chain, chainBad, ok ? "OK" : "FAIL");
  if (!out.empty()) {
    std::vector<uint8_t> both((size_t)w * 2 * h * 4);
    for (int y = 0; y < h; ++y) {
      memcpy(&both[(size_t)y * 2 * w * 4], &preview[(size_t)y * w * 4], (size_t)w * 4);
      memcpy(&both[((size_t)y * 2 * w + w) * 4], &direct[(size_t)y * w * 4], (size_t)w * 4);
    }
    writePNG(out.c_str(), both.data(), 2 * w, h);
    printf("wrote %s (preview left, direct render right)\n", out.c_str());
  }
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------- headless --

int testFlicker(const Options& o, double factor);
int testHalo(const Options& o, double factor, int steps);

int runHeadless(const Options& o) {
  CUDA_CHECK(cudaSetDevice(0));
  if (o.flickerTest) return testFlicker(o, o.zoomFactor);
  if (o.haloTest) return testHalo(o, o.zoomFactor, o.haloSteps);
  ViewParams v = o.view;
  if (!o.iterGiven) v.maxIter = autoIterations(defaultScale(o.width) / v.scale);
  const int w = o.width, h = o.height;
  const size_t bytes = (size_t)w * h * 4;
  Accum accum = allocAccum(w, h, totalSamples(v), wantsDouble(v));
  uchar4* dev = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, bytes));
  if (o.resumeTest) return testResume(accum, dev, w, h, v);
  if (o.recolorTest) return testRecolor(accum, dev, w, h, v);
  if (o.deeperTest) {
    uchar4* dev2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dev2, bytes));
    return testDeeper(accum, dev, dev2, w, h, v);
  }
  if (o.zoomTest) {
    uchar4* dev2 = nullptr;
    CUDA_CHECK(cudaMalloc(&dev2, bytes));
    return testZoom(accum, dev, dev2, w, h, v, o.zoomFactor, o.shiftDx, o.shiftDy, o.headlessOut);
  }
  if (o.shiftTest) {
    Accum accum2 = allocAccum(w, h, totalSamples(v), wantsDouble(v));
    return testShift(accum, accum2, dev, w, h, v, o.shiftDx, o.shiftDy);
  }

  RenderStats st = renderAllSamples(accum, w, h, v);
  const int ns = totalSamples(v);
  if (o.benchFrames > 0) {
    double total = 0;
    for (int i = 0; i < o.benchFrames; ++i) total += renderAllSamples(accum, w, h, v).kernelMs;
    printf("%d frames of %dx%d, %d iterations, %d samples/pixel, %s: %.3f ms/frame (%.3f ms/sample)\n",
           o.benchFrames, w, h, v.maxIter, ns, st.usedDouble ? "double" : "float", total / o.benchFrames,
           total / o.benchFrames / ns);
  } else {
    printf("rendered %dx%d, %d iterations, %d samples/pixel, %s precision, kernel %.3f ms\n", w, h, v.maxIter,
           ns, st.usedDouble ? "double" : "float", st.kernelMs);
  }
  if (!o.headlessOut.empty()) {
    resolveSamples(accum, nullptr, 0.0f, dev, w, h);
    std::vector<uint8_t> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost));
    if (!writePNG(o.headlessOut.c_str(), host.data(), w, h)) {
      fprintf(stderr, "failed to write %s\n", o.headlessOut.c_str());
      return 1;
    }
    printf("wrote %s\n", o.headlessOut.c_str());
  }
  cudaFree(dev);
  freeAccum(accum);
  return 0;
}

// --------------------------------------------------------------------- GUI --

// The GL-free rendering state machine: the view, the accumulation buffers, the image on
// screen and the pass scheduling. The GUI (App) wraps it; --test-flicker drives it headless.
struct Renderer {
  int w = 0, h = 0;  // image size in pixels
  int maxIterCap = kMaxIter;  // the effective limit never exceeds this (the self-tests lower it)
  int iterBudget = kIterBudget;  // iterations per sample and pass at most (lowered after slow frames)
  bool coarseChase = true;  // the coarse image may run ahead to the preview's depth (not after a zoom in)
  int firstSamples = 2;     // samples per pixel the first pass of this view takes (planFirstPass)
  float plannedMs = 0;      // predicted cost of that first pass
  double plannedWork = 0;   // the modelled iterations of that first pass (calibrates the rate)
  double rateFloat = 0, rateDouble = 0;  // iterations per ms a full pass achieves, learned from first passes
  Accum accum;                         // sample accumulation buffer
  Accum accum2;                        // scratch for translating the accumulation buffer
  uchar4* display = nullptr;           // the image currently shown
  uchar4* preview = nullptr;           // last displayed image resampled into the current view
  // The surround: a half-resolution, low-iteration image of a field kSurroundField times wider
  // than the view, rendered when the view leaves the field it covers. The border a zoom out
  // or a pan uncovers shows it (as a prior trusted only to its depth) instead of black until
  // real samples arrive.
  uchar4* surround = nullptr;
  int sw = 0, sh = 0;                  // its size
  ViewParams surroundView;             // the view it shows (maxIter: the limit it was computed with)
  bool surroundValid = false;
  float surroundMs = 0;                // kernel time of the last surround render
  Accum surroundAccum;

  ViewParams view;
  ViewParams homeView;
  ViewParams shownView;     // the view whose content `display` holds
  int displayUseful = 0;    // depth the content of `display` needed: where a zoom in starts (see usefulIter)
  int displayVerified = 0;  // depth the colors of `display` were tested to: black there is black to this depth
  double displayMag = 1;    // how much the colors of `display` have been magnified since they were computed
  int usefulIter = 0;       // limit of the last pass that rescued a noticeable share of pixels, plus a step
  int previewVerified = 0;  // displayVerified of the image the preview was made from
  double previewMag = 1;    // magnification of the preview's colors since they were computed
  bool trace = getenv("JF_TRACE") != nullptr;  // per-pass statistics on stderr
  int previewIter = 0;      // depth the preview's colors stand in for black pixels to (0: not trusted)
  int sampleIter = 0;       // iteration limit of the latest pass
  bool autoIter = true;
  bool dirty = true;        // view changed: restart sampling from scratch
  bool appearanceChanged = false;  // iterations/precision changed: old colors are no prior
  bool recolor = false;     // palette/color scale/offset changed: recolor the kept samples
  bool recolored = false;   // a recolor was applied and must be shown
  float previewWeight = 0;  // how many samples the current preview counts for
  int iterAdjust = 0;       // extra iterations added by progressive deepening (see kIterStep)
  int pendingShiftX = 0;    // pan in whole pixels not yet applied to the accumulation buffer
  int pendingShiftY = 0;
  double dragRemX = 0, dragRemY = 0;  // sub-pixel drag remainders, so pans stay on the pixel grid
  int passes = 0;           // passes since the last restart or shift
  bool depthReached = false;  // the single-sample deepening has settled: antialiasing samples are being added
  int aaStartPass = 0;      // pass at which the antialiasing samples started
  bool aaDone = false;      // every pixel has all its samples
  bool firstPassIsFull = false;  // next pass computes every pixel (after a clear)
  bool previewPending = false;   // a change was applied but nothing of it shown yet
  float fullPassMs = 0;     // cost of the last pass that computed every pixel
  float frameMs = 0;        // kernel time spent on the current view so far
  RenderStats stats;

  double zoom() const { return defaultScale(w) / view.scale; }
  int effectiveIter() const { return std::min(view.maxIter + iterAdjust, maxIterCap); }
  // Samples per pixel allowed so far: two in the first pass (the second only if the first
  // escaped), doubling in every early pass at the low limit (4, 8, 16; see earlyAA), one per
  // pixel while deepening, then doubling again per pass for the pixels still missing them.
  // Counts stay powers of two, so a pixel's mean color is an exact division.
  int earlyPasses() const {
    int n = 0;
    while ((firstSamples << n) <= totalSamples(view)) ++n;
    return n;
  }
  bool earlyAA() const { return passes < earlyPasses(); }
  int samplesActive() const {
    if (earlyAA()) return std::min(totalSamples(view), firstSamples << passes);
    return depthReached ? std::min(totalSamples(view), 2 << (passes - aaStartPass)) : 1;
  }
  bool converged() const { return aaDone; }
  // Nothing left to do for this view: all samples taken and (almost) no pixel still hits the
  // limit, or the limit cannot grow any further.
  // Nothing left to do for this view: all samples taken and (almost) no pixel still hits the
  // limit, or the limit cannot grow any further. While pixels keep the colors of a deeper
  // preview, deepening goes on at least until the limit has reached the preview's, so those
  // colors are either confirmed by an escape or replaced by black.
  bool done() const {
    if (!converged()) return false;
    if (effectiveIter() >= maxIterCap || stats.hitCount == 0) return true;
    return stats.hitCount < kFewHits && effectiveIter() >= previewIter;
  }
  bool busy() const { return dirty || pendingShiftX || pendingShiftY || !done(); }

  void pixelToWorld(double px, double py, double& wx, double& wy) const {
    wx = view.centerX + (px - 0.5 * w) * view.scale;
    wy = view.centerY - (py - 0.5 * h) * view.scale;
  }

  void freeFrame() {
    if (display) {
      cudaFree(display);
      display = nullptr;
    }
    if (preview) {
      cudaFree(preview);
      preview = nullptr;
    }
    if (surround) {
      cudaFree(surround);
      surround = nullptr;
    }
    freeAccum(accum);
    freeAccum(accum2);
    freeAccum(surroundAccum);
    surroundValid = false;
  }

  // Renders the surround for the current view: one sample per pixel at a low limit, which a
  // single pass completes. Cheap (a quarter of the pixels, 1000 iterations) next to a first
  // pass of the view itself.
  void renderSurround() {
    ViewParams s = view;
    s.samples = 1;
    s.scale = view.scale * kSurroundField * (double)w / sw;
    s.maxIter = std::min(std::max(kSurroundIter, autoIterations(defaultScale(w) / s.scale)), iterBudget);
    ensureOrbits(surroundAccum, sw, sh, 1, wantsDouble(s));
    clearSamples(surroundAccum, sw, sh);
    surroundMs = renderPass(surroundAccum, sw, sh, s, nullptr, 0, 0, 0, false, 0, iterBudget).kernelMs;
    resolveSamples(surroundAccum, nullptr, 0.0f, surround, sw, sh);
    surroundView = s;
    surroundValid = true;
  }
  // True when the surround still has room around `v`: `v` lies inside its field with half the
  // field to spare (so the next zoom out is covered too) and is not so small that the
  // surround would be a blur there.
  bool surroundCovers(const ViewParams& v) const {
    if (!surroundValid) return false;
    const double fieldX = surroundView.scale * sw, fieldY = surroundView.scale * sh;
    const double extentX = v.scale * w, extentY = v.scale * h;
    if (extentX > 0.5 * fieldX || extentX < fieldX / 16) return false;
    return std::abs(v.centerX - surroundView.centerX) + 0.5 * extentX <= 0.5 * fieldX &&
           std::abs(v.centerY - surroundView.centerY) + 0.5 * extentY <= 0.5 * fieldY;
  }
  int shallowIter() const { return surroundValid ? surroundView.maxIter : 0; }

  // Probes a new view with the coarse 4x4 field before its first pass and plans that pass to
  // stay within kFirstPassTargetMs. The field is taken in stages from the auto limit `d0`,
  // doubling up to the inherited depth `d1`, until the probing has used kProbeBudgetMs. At
  // each complete stage the first pass's cost is predicted from the field: the 16 pixels of
  // an escaped point cost about what the point took, per sample; those around a boundary
  // point (one with an escaped neighbour) one sample at the limit, capped by the per-pass
  // budget (the pass stops at a pixel's first hit); blocked interior pixels nothing. The rate
  // is the one the last planned first pass achieved in this precision (the probe's own rate
  // until then, which runs far fewer threads and so errs on the slow side). The deepest stage
  // that fits with the two primary samples wins, failing that the shallowest with one sample,
  // and the sample count then doubles while it still fits. The field stays for the passes,
  // which block with it from the start.
  void planFirstPass(int d0, int d1) {
    const int total = totalSamples(view);
    ViewParams v = view;
    double probeMs = 0;
    int bestDepth = d0, bestSamples = 1;
    plannedMs = 0;
    bool planned = false;
    for (int d = d0;; d = std::min(d * 2, d1)) {
      v.maxIter = d;
      CoarseStats st;
      double stageMs = 0;
      do {
        st = coarsePass(accum, w, h, v, d, kCoarseBudgetFactor * iterBudget);
        stageMs += st.kernelMs;
      } while (!st.complete && probeMs + stageMs < kProbeBudgetMs);
      probeMs += stageMs;
      if (!st.complete) break;  // out of probing time: the previous stage stands
      const double learned = wantsDouble(view) ? rateDouble : rateFloat;
      const double rate = learned > 0 ? learned : st.work > 0 ? st.work / std::max(probeMs, 1e-3) : 0;
      auto work = [&](int n) {
        return 16.0 * (n * st.escapedIters + (double)st.boundary * std::min(d, iterBudget));
      };
      auto predict = [&](int n) { return rate > 0 ? work(n) / rate : 0.0; };
      const double two = predict(2), one = predict(1);
      if (trace)
        fprintf(stderr, "probe depth %d: %d hits (%d boundary) of %d points, probe %.2f ms so far, first pass "
                "predicted %.1f ms with 2 samples\n", d, st.hits, st.boundary, st.points, probeMs, two);
      if (two <= kFirstPassTargetMs) {
        bestDepth = d;
        bestSamples = 2;
        plannedMs = (float)two;
        plannedWork = work(2);
        while (bestSamples * 2 <= total && predict(bestSamples * 2) <= kFirstPassTargetMs) {
          bestSamples *= 2;
          plannedMs = (float)predict(bestSamples);
          plannedWork = work(bestSamples);
        }
        planned = true;
      } else {
        if (!planned) {  // not even the auto limit with two samples fits: one sample there
          bestDepth = d;
          bestSamples = 1;
          plannedMs = (float)one;
          plannedWork = work(1);
          planned = true;
        }
        break;  // deeper only costs more
      }
      if (d >= d1) break;
    }
    iterAdjust = std::max(0, bestDepth - view.maxIter);
    firstSamples = std::min(bestSamples, total);
    if (trace)
      fprintf(stderr, "first pass: limit %d, %d samples, predicted %.1f ms (probe %.2f ms)\n", bestDepth,
              firstSamples, plannedMs, probeMs);
  }

  // Called with the real time an update took: a slow one (over kSlowFrameMs) cuts the budget
  // of the next passes in proportion, so the next update does not take that long again; fast
  // ones restore it gradually.
  void adaptBudget(double wallMs) {
    if (wallMs > kSlowFrameMs)
      iterBudget = std::max(kMinIterBudget, (int)(iterBudget * kFrameTargetMs / wallMs));
    else if (wallMs < 0.5 * kFrameTargetMs && iterBudget < kIterBudget)
      iterBudget = std::min(kIterBudget, iterBudget * 2);
  }

  // Allocates the buffers for the current w x h; the view starts from scratch.
  void allocFrame() {
    freeFrame();
    const size_t bytes = (size_t)w * h * 4;
    accum = allocAccum(w, h, totalSamples(view), wantsDouble(view));
    accum2 = allocAccum(w, h, totalSamples(view), wantsDouble(view));
    CUDA_CHECK(cudaMalloc(&display, bytes));
    CUDA_CHECK(cudaMalloc(&preview, bytes));
    CUDA_CHECK(cudaMemset(display, 0, bytes));
    sw = std::max(w / 2, 1);
    sh = std::max(h / 2, 1);
    surroundAccum = allocAccum(sw, sh, 1, wantsDouble(view));
    CUDA_CHECK(cudaMalloc(&surround, (size_t)sw * sh * 4));
    surroundValid = false;
    shownView = view;
    dirty = true;
  }


  // Applies whatever changed since the last frame: a full restart, or a translation that
  // keeps the samples of every pixel still on screen.
  void applyChanges() {
    if (recolor) {
      recolor = false;
      if (!dirty) {
        // No geometry change: keep every sample and orbit, just rebuild the colors.
        recolorSamples(accum, w, h, view);
        previewWeight = 0;  // the old colors are no prior any more
        previewIter = 0;
        previewVerified = 0;
        recolored = true;
        shownView = view;
      }
    }
    if (!dirty && !pendingShiftX && !pendingShiftY) return;
    // Whatever is on screen, resampled into the new view, stands in until real samples arrive.
    resamplePreview(display, preview, w, h, shownView, view);
    // The border a zoom out or a pan uncovers shows the low-iteration surround image instead
    // of black, so nothing flickers to black between the zooms of a sequence.
    if (!surroundCovers(view)) renderSurround();
    fillPreview(surround, sw, sh, surroundView, preview, w, h, view);
    // The preview's colors keep standing in for black pixels, to the depth they were tested
    // to, as long as they are still valid and not magnified beyond kPreviewTrust in total.
    previewMag = displayMag * (shownView.scale / view.scale);
    previewVerified = appearanceChanged ? 0 : displayVerified;
    previewIter = (appearanceChanged || previewMag > kPreviewTrust) ? 0 : previewVerified;
    // Geometric changes keep the old image as a prior; appearance changes make its colors stale.
    previewWeight = appearanceChanged ? 0.0f : kPreviewWeight;
    const bool appearance = appearanceChanged;
    appearanceChanged = false;
    if (dirty) {
      dirty = false;
      // A zoom in starts at the depth the image on screen needed (never shallower than what
      // is shown), so nothing resets to a lower limit and the preview never has to hold
      // colors for black pixels. A zoom out starts at the default limit instead: the border
      // it uncovers has no preview and should be drawn quickly, while the covered area keeps
      // its deeper colors through the preview until the deepening has caught up. Iteration
      // keys reset the adjustment themselves; other appearance changes keep the depth.
      const bool zoomingOut = view.scale > shownView.scale;
      // After a zoom in the coarse field starts clean and follows the new view's limit only:
      // it does not chase the previous view's depth, so nothing of that computation
      // propagates into the new view. A zoom out keeps the chase, which clears the covered
      // area's stale preview colors at the depth they were tested to.
      coarseChase = zoomingOut;
      const int keep = appearance ? effectiveIter() : zoomingOut ? 0 : displayUseful;
      // The planner may go beyond the useful depth, with headroom, if the previous image
      // verified that deep and the time budget allows.
      const int reach = appearance || zoomingOut
                            ? keep
                            : std::min(displayVerified, (int)std::min<long long>(4LL * kDepthHeadroom * displayUseful / 4, maxIterCap));
      if (autoIter) view.maxIter = autoIterations(zoom());
      iterAdjust = std::max(0, std::min(keep, maxIterCap) - view.maxIter);
      ensureOrbits(accum, w, h, totalSamples(view), wantsDouble(view));
      ensureOrbits(accum2, w, h, totalSamples(view), wantsDouble(view));
      clearSamples(accum, w, h);
      firstPassIsFull = true;
      firstSamples = 2;
      plannedMs = 0;
      // A zoom plans its first pass with the coarse field: the depth it starts at (between the
      // auto limit and the inherited depth) and the samples it takes, to a time budget.
      if (!appearance) planFirstPass(view.maxIter, std::max(view.maxIter, std::min(reach, maxIterCap)));
    } else {
      shiftSamples(accum, accum2, w, h, pendingShiftX, pendingShiftY);
      std::swap(accum, accum2);
      firstPassIsFull = false;
      coarseChase = true;
      firstSamples = 2;
    }
    pendingShiftX = pendingShiftY = 0;
    passes = 0;
    depthReached = aaDone = false;
    aaStartPass = 0;
    sampleIter = 0;  // no previous pass on this view yet
    frameMs = 0;
    previewPending = true;
    shownView = view;
  }

  // True when the first real pass would take long enough that the preview is worth a frame:
  // by the plan's prediction when there is one, else by the last full pass's cost.
  bool previewWorthShowing() const {
    return firstPassIsFull && (plannedMs > 0 ? plannedMs : fullPassMs) > kPreviewThresholdMs;
  }

  // Moves the view by whole pixels; the accumulation buffer is translated to match.
  void pan(int dx, int dy) {
    if (!dx && !dy) return;
    view.centerX += dx * view.scale;
    view.centerY -= dy * view.scale;
    pendingShiftX += dx;
    pendingShiftY += dy;
  }

  // Accumulates samples until converged or the kernel-time budget is used up.
  void renderPasses(double budgetMs) {
    double spent = 0;
    do {
      ViewParams v = view;
      v.maxIter = effectiveIter();
      const int active = samplesActive();
      const bool deepened = v.maxIter > sampleIter;  // this pass went deeper than the last one
      const bool early = earlyAA();
      stats = renderPass(accum, w, h, v, previewWeight > 0 ? preview : nullptr, previewIter, sampleIter, active,
                         early, shallowIter(), iterBudget, coarseChase);
      sampleIter = v.maxIter;
      if (depthReached && active >= totalSamples(view)) aaDone = true;
      // The depth this view really needed: beyond the last pass that still rescued a
      // noticeable share of the probed pixels (boundary pixels trickle in at every depth),
      // deepening changes next to nothing, so that is what the next view starts from, not
      // the cap we chase afterwards.
      if (passes == 0) usefulIter = v.maxIter;
      if (stats.rescueCount >= std::max(kFewHits, stats.hitCount / 100))
        usefulIter = std::max(usefulIter, stats.maxRescueIter + kIterStep);
      if (trace)
        fprintf(stderr, "pass %d limit %d samples %d hits %d rescues %d (deepest escape %d) exposed %d %.3f ms\n",
                passes, v.maxIter, active, stats.hitCount, stats.rescueCount, stats.maxRescueIter,
                stats.exposedCount, stats.kernelMs);
      if (firstPassIsFull) {
        fullPassMs = stats.kernelMs;
        // Calibrate the planner's rate with what this first pass really achieved.
        if (plannedWork > 0 && stats.kernelMs > 0.05f)
          (stats.usedDouble ? rateDouble : rateFloat) = plannedWork / stats.kernelMs;
        plannedWork = 0;
      }
      firstPassIsFull = false;
      ++passes;
      // Depth first, with one sample per pixel: the antialiasing samples start once the
      // deepening has settled, i.e. nothing is being exposed and (almost) nothing is rescued
      // any more, or the limit has caught up with a trusted preview, or it cannot grow.
      if (!depthReached && passes > earlyPasses() && stats.exposedCount == 0) {
        const bool trickle = deepened && stats.rescueCount < kFewHits;  // a held limit rescues nothing anyway
        const bool caughtUp = previewIter > 0 && effectiveIter() >= previewIter &&
                              stats.rescueCount < std::max(kFewHits, stats.hitCount / 100);
        if (stats.hitCount < kFewHits || trickle || caughtUp || effectiveIter() >= maxIterCap) {
          depthReached = true;
          aaStartPass = passes;
        }
      }
      // Some pixel ran to the limit: give the next pass more iterations to try to resolve it.
      // Not while the black region is still eroding, though: pixels it just exposed first
      // catch up at the current limit before anyone goes deeper. And not while the
      // antialiasing samples are being added: they all land at the settled depth.
      const bool addingSamples = depthReached && !aaDone;
      if (stats.hitCount > 0 && stats.exposedCount == 0 && !addingSamples) iterAdjust += kIterStep;
      spent += stats.kernelMs;
      frameMs += stats.kernelMs;
    } while (!done() && spent < budgetMs);
  }

  // Resolves the accumulated samples (and preview, where samples are missing) into `display`
  // and keeps track of what the screen now shows.
  void resolveDisplay() {
    previewPending = false;
    resolveSamples(accum, preview, priorWeight(previewWeight, passes, totalSamples(view)), display, w, h,
                   previewIter, sampleIter, shallowIter());
    // What is on screen now: with passes, the samples (tested to the latest limit) plus, where
    // black pixels still show trusted preview colors, those colors at the preview's depth and
    // magnification; without passes, the preview itself.
    const bool previewShowing = previewIter > sampleIter;  // the deeper-preview rule still acts
    displayUseful = passes > 0 ? usefulIter : displayUseful;
    displayVerified = passes > 0 ? std::max(sampleIter, previewShowing ? previewIter : 0) : previewVerified;
    displayMag = (passes > 0 && !previewShowing) ? 1.0 : previewMag;
  }

  void zoomAt(double px, double py, double factor) {
    double wx, wy;
    pixelToWorld(px, py, wx, wy);
    view.scale /= factor;
    view.centerX = wx + (view.centerX - wx) / factor;
    view.centerY = wy + (view.centerY - wy) / factor;
    dirty = true;
  }
};

struct App : Renderer {
  SDL_Window* win = nullptr;
  SDL_GLContext gl = nullptr;
  GLuint pbo = 0, tex = 0;
  cudaGraphicsResource* cudaPbo = nullptr;
  bool interop = true;                 // false: copy through host memory instead
  std::vector<uint8_t> hostPixels;     // used for fallback path and screenshots
  bool fullscreen = false;
  bool dragging = false;
  int lastMouseX = 0, lastMouseY = 0;

  // Window (logical) coordinates -> drawable pixel coordinates.
  void mouseToPixel(int mx, int my, double& px, double& py) const {
    int ww, wh;
    SDL_GetWindowSize(win, &ww, &wh);
    px = mx * (double)w / std::max(ww, 1);
    py = my * (double)h / std::max(wh, 1);
  }

  void destroyBuffers() {
    if (cudaPbo) {
      cudaGraphicsUnregisterResource(cudaPbo);
      cudaPbo = nullptr;
    }
    freeFrame();
    if (pbo) pglDeleteBuffers(1, &pbo), pbo = 0;
    if (tex) glDeleteTextures(1, &tex), tex = 0;
  }

  void createBuffers() {
    destroyBuffers();
    SDL_GL_GetDrawableSize(win, &w, &h);
    w = std::max(w, 1);
    h = std::max(h, 1);
    size_t bytes = (size_t)w * h * 4;

    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    if (interop) {
      pglGenBuffers(1, &pbo);
      pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
      pglBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_STREAM_DRAW);
      pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA/GL interop unavailable (%s); falling back to host copies.\n",
                cudaGetErrorString(err));
        cudaGetLastError();
        interop = false;
        pglDeleteBuffers(1, &pbo);
        pbo = 0;
      }
    }
    if (!interop) hostPixels.resize(bytes);
    allocFrame();
  }

  // Uploads the resolved image to the texture.
  void resolveToTexture() {
    resolveDisplay();
    const size_t bytes = (size_t)w * h * 4;
    glBindTexture(GL_TEXTURE_2D, tex);
    if (interop) {
      uchar4* dev = nullptr;
      size_t size = 0;
      CUDA_CHECK(cudaGraphicsMapResources(1, &cudaPbo, 0));
      CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dev, &size, cudaPbo));
      CUDA_CHECK(cudaMemcpy(dev, display, bytes, cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaPbo, 0));
      pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
      pglBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    } else {
      CUDA_CHECK(cudaMemcpy(hostPixels.data(), display, bytes, cudaMemcpyDeviceToHost));
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, hostPixels.data());
    }
  }

  void present() {
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, 1);   // texture row 0 is the top of the image
    glTexCoord2f(1, 0); glVertex2f(1, 1);
    glTexCoord2f(1, 1); glVertex2f(1, -1);
    glTexCoord2f(0, 1); glVertex2f(-1, -1);
    glEnd();
    SDL_GL_SwapWindow(win);
  }

  void updateTitle() {
    const char* prec = view.precision == Precision::Auto ? (stats.usedDouble ? "auto:double" : "auto:float")
                       : view.precision == Precision::Float ? "float" : "double";
    char buf[512];
    snprintf(buf, sizeof buf,
             "JyrkisFractals | Mandelbrot | zoom %.4g | iter %d+%d%s | %s | AA %d/%d%s | hits %d | render %.1f ms%s | "
             "palette %d | center %.17g %+.17gi",
             zoom(), view.maxIter, iterAdjust, autoIter ? " (auto)" : "", prec,
             passes ? samplesActive() : 0, totalSamples(view),
             passes == 0                                                 ? " (preview)"
             : earlyAA()                                                 ? " (early)"
             : !depthReached                                             ? " (deepening)"
             : priorWeight(previewWeight, passes, totalSamples(view)) > 0 ? " (+preview prior)"
                                                                         : "",
             passes ? stats.hitCount : 0,
             frameMs, iterBudget < kIterBudget ? " (budget cut)" : "", view.palette,
             view.centerX, view.centerY);
    SDL_SetWindowTitle(win, buf);
  }

  void saveScreenshot(std::string name = "") {
    applyChanges();
    while (!converged()) renderPasses(1e9);  // finish antialiasing first (deepening may go on)
    resolveToTexture();
    present();
    updateTitle();
    size_t bytes = (size_t)w * h * 4;
    hostPixels.resize(bytes);
    CUDA_CHECK(cudaMemcpy(hostPixels.data(), display, bytes, cudaMemcpyDeviceToHost));
    if (name.empty()) name = timestampName(".png");
    if (writePNG(name.c_str(), hostPixels.data(), w, h))
      printf("saved %s  (%dx%d, center %.17g %+.17gi, scale %.17g, iter %d, samples %d, palette %d)\n",
             name.c_str(), w, h, view.centerX, view.centerY, view.scale, effectiveIter(), totalSamples(view),
             view.palette);
    else
      fprintf(stderr, "failed to write %s\n", name.c_str());
  }

  void handleKey(const SDL_KeyboardEvent& k) {
    const bool shift = k.keysym.mod & KMOD_SHIFT;
    const double panStep = 0.1;
    const SDL_Keycode sym = k.keysym.sym;
    if (sym >= SDLK_0 && sym <= SDLK_9) {  // samples per pixel: 1..9 rounded up to a power of two, 0: 16
      view.samples = sym == SDLK_0 ? kMaxSamples : sym - SDLK_0;
      dirty = true;
      return;
    }
    switch (sym) {  // changes that alter colors rather than geometry
      case SDLK_a: case SDLK_PLUS: case SDLK_EQUALS: case SDLK_KP_PLUS: case SDLK_MINUS: case SDLK_KP_MINUS:
      case SDLK_d:
        appearanceChanged = true;
        break;
      default: break;
    }
    switch (sym) {
      case SDLK_ESCAPE:
      case SDLK_q: SDL_Event q; q.type = SDL_QUIT; SDL_PushEvent(&q); break;
      case SDLK_r: {
        int samples = view.samples;
        view = homeView;
        view.scale = defaultScale(w);
        view.samples = samples;
        autoIter = true;
        dirty = true;
        break;
      }
      case SDLK_h: printf("%s", kHelp); break;
      case SDLK_f:
        fullscreen = !fullscreen;
        SDL_SetWindowFullscreen(win, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
        break;
      case SDLK_s: saveScreenshot(); break;
      case SDLK_a: autoIter = !autoIter; iterAdjust = 0; dirty = true; break;
      case SDLK_PLUS:
      case SDLK_EQUALS:
      case SDLK_KP_PLUS:
        autoIter = false; view.maxIter = std::min(effectiveIter() * 3 / 2, kMaxIter); iterAdjust = 0; dirty = true;
        break;
      case SDLK_MINUS:
      case SDLK_KP_MINUS:
        autoIter = false; view.maxIter = std::max(effectiveIter() * 2 / 3, 16); iterAdjust = 0; dirty = true;
        break;
      case SDLK_d:
        view.precision = (Precision)(((int)view.precision + 1) % 3);
        dirty = true;
        break;
      // Color-only changes: the escape values are kept, the samples are just recolored.
      case SDLK_p:
        view.palette = (view.palette + (shift ? kPaletteCount - 1 : 1)) % kPaletteCount;
        recolor = true;
        break;
      case SDLK_LEFTBRACKET: view.colorScale *= 1.25f; recolor = true; break;
      case SDLK_RIGHTBRACKET: view.colorScale /= 1.25f; recolor = true; break;
      case SDLK_COMMA: view.colorOffset -= 0.02f; recolor = true; break;
      case SDLK_PERIOD: view.colorOffset += 0.02f; recolor = true; break;
      case SDLK_z: zoomAt(0.5 * w, 0.5 * h, 1.5); break;
      case SDLK_x: zoomAt(0.5 * w, 0.5 * h, 1.0 / 1.5); break;
      case SDLK_PAGEUP:
      case SDLK_PAGEDOWN: {
        // Zoom about the mouse position (the window center if the pointer is outside).
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        double px, py;
        mouseToPixel(mx, my, px, py);
        if (!(SDL_GetWindowFlags(win) & SDL_WINDOW_MOUSE_FOCUS)) px = 0.5 * w, py = 0.5 * h;
        zoomAt(px, py, sym == SDLK_PAGEUP ? 1.5 : 1.0 / 1.5);
        break;
      }
      case SDLK_LEFT: pan(-(int)std::lround(panStep * w), 0); break;
      case SDLK_RIGHT: pan((int)std::lround(panStep * w), 0); break;
      case SDLK_UP: pan(0, -(int)std::lround(panStep * h)); break;
      case SDLK_DOWN: pan(0, (int)std::lround(panStep * h)); break;
      default: break;
    }
  }

  // Returns false when the application should quit.
  bool handleEvent(const SDL_Event& e) {
    switch (e.type) {
      case SDL_QUIT: return false;
      case SDL_WINDOWEVENT:
        if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) createBuffers();
        if (e.window.event == SDL_WINDOWEVENT_EXPOSED && !busy() && passes > 0) present();
        break;
      case SDL_KEYDOWN: handleKey(e.key); break;
      case SDL_MOUSEWHEEL: {
        int mx, my;
        SDL_GetMouseState(&mx, &my);
        double px, py;
        mouseToPixel(mx, my, px, py);
        float steps = e.wheel.preciseY != 0 ? e.wheel.preciseY : (float)e.wheel.y;
        if (e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) steps = -steps;
        if (steps != 0) zoomAt(px, py, std::pow(1.25, steps));
        break;
      }
      case SDL_MOUSEBUTTONDOWN: {
        double px, py;
        mouseToPixel(e.button.x, e.button.y, px, py);
        if (e.button.button == SDL_BUTTON_LEFT) {
          if (e.button.clicks == 2) zoomAt(px, py, 2.0);
          dragging = true;
          dragRemX = dragRemY = 0;
          lastMouseX = e.button.x;
          lastMouseY = e.button.y;
        } else if (e.button.button == SDL_BUTTON_RIGHT) {
          zoomAt(px, py, 0.5);
        }
        break;
      }
      case SDL_MOUSEBUTTONUP:
        if (e.button.button == SDL_BUTTON_LEFT) dragging = false;
        break;
      case SDL_MOUSEMOTION:
        if (dragging && (e.motion.state & SDL_BUTTON_LMASK)) {
          double ax, ay, bx, by;
          mouseToPixel(lastMouseX, lastMouseY, ax, ay);
          mouseToPixel(e.motion.x, e.motion.y, bx, by);
          // Dragging the image right moves the view center left. Keep pans on whole pixels
          // so the already computed samples can be reused; carry the remainder.
          dragRemX += ax - bx;
          dragRemY += ay - by;
          int dx = (int)std::lround(dragRemX), dy = (int)std::lround(dragRemY);
          dragRemX -= dx;
          dragRemY -= dy;
          pan(dx, dy);
          lastMouseX = e.motion.x;
          lastMouseY = e.motion.y;
        }
        break;
      default: break;
    }
    return true;
  }
};

// Self-test for the zoom schedule: converges a view exactly as the GUI does (one pass per
// frame), zooms by `factor` about the center (less than 1: out), and watches every frame until
// the new view has converged too. Black must not appear temporarily: a pixel that the preview
// showed colored and that ends up colored must never be black by rule (all samples hit, or
// blocked by the coarse image) in between; an exact mean of near-black palette colors may
// round to zero and does not count. Colors the preview
// held as trusted (alpha kPreviewAlphaTrusted) must never flash at all; colors from the
// shallow surround fill (the border of a zoom out) are approximate, so a small share may.
int testFlicker(const Options& o, double factor) {
  Renderer r;
  r.w = o.width;
  r.h = o.height;
  r.view = o.view;
  r.autoIter = !o.iterGiven;
  r.maxIterCap = kFlickerTestCap;
  r.allocFrame();
  const size_t n = (size_t)r.w * r.h, bytes = n * 4;
  std::vector<uint8_t> prior(bytes), frame(bytes);
  auto grab = [&](std::vector<uint8_t>& dst) {
    r.resolveDisplay();
    CUDA_CHECK(cudaMemcpy(dst.data(), r.display, bytes, cudaMemcpyDeviceToHost));
  };
  auto colored = [](const std::vector<uint8_t>& img, size_t i) {
    return (img[4 * i] | img[4 * i + 1] | img[4 * i + 2]) != 0;
  };
  r.applyChanges();
  int passesA = 0;
  do {
    r.renderPasses(0);  // one pass
    ++passesA;
  } while (!r.done());
  grab(frame);
  r.zoomAt(0.5 * r.w, 0.5 * r.h, factor);
  r.applyChanges();
  grab(prior);  // the preview frame: what the screen shows right after the zoom
  const float surroundMs = r.surroundMs;
  std::vector<int> firstBlack(n, 0);  // pass at which the pixel first showed black (0: never)
  std::vector<int> blackFrames(n, 0);  // how many frames the pixel showed black by rule
  std::vector<float> passWeights;
  std::vector<unsigned short> passMeta;
  int passesB = 0;
  int px = -1, py = -1;  // JF_FLICKER_PIXEL="x y": trace this pixel through the passes
  if (const char* e = getenv("JF_FLICKER_PIXEL")) sscanf(e, "%d %d", &px, &py);
  do {
    r.renderPasses(0);
    ++passesB;
    grab(frame);
    // Black by rule, not by color: a pixel whose samples all hit (or that the coarse image
    // blocked) and that shows black. An exact mean of near-black palette colors and hit
    // samples can round to zero too, and that is no flash.
    readCounts(r.accum, r.w, r.h, passWeights, passMeta);
    for (size_t i = 0; i < n; ++i)
      if (!colored(frame, i) && showsBlack(passMeta[i])) {
        if (!firstBlack[i]) firstBlack[i] = passesB;
        ++blackFrames[i];
      }
    if (px >= 0 && passesB <= 16) {
      std::vector<float> wts;
      std::vector<unsigned short> mt;
      readCounts(r.accum, r.w, r.h, wts, mt);
      const size_t i = (size_t)py * r.w + px;
      uchar4 pv;
      CUDA_CHECK(cudaMemcpy(&pv, r.preview + i, 4, cudaMemcpyDeviceToHost));
      fprintf(stderr, "pass %d limit %d: pixel (%d, %d) shows %d %d %d, preview %d %d %d a%d, weight %.3f, taken %d hits %d%s%s, previewIter %d\n",
              passesB, r.sampleIter, px, py, frame[4 * i], frame[4 * i + 1], frame[4 * i + 2], pv.x, pv.y, pv.z, pv.w,
              wts[i], samplesTaken(mt[i]), limitHits(mt[i]), isCertain(mt[i]) ? " certain" : "",
              isCoarseBlack(mt[i]) ? " coarse" : "", r.previewIter);
    }
  } while (!r.done());
  size_t trusted = 0, shallow = 0, trustedEnd = 0, shallowEnd = 0, trustedFlash = 0, shallowFlash = 0,
         shallowBlackFrames = 0;
  std::vector<size_t> flashAt(passesB + 1, 0);
  const bool dump = getenv("JF_FLICKER_DUMP") != nullptr;  // print some of the flashing pixels
  int dumped = 0;
  std::vector<float> weights;
  std::vector<unsigned short> meta;
  readCounts(r.accum, r.w, r.h, weights, meta);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t a = prior[4 * i + 3];
    if (a == 0 || !colored(prior, i)) continue;
    const bool isTrusted = a == kPreviewAlphaTrusted;  // else shallow surround or mixed boundary color
    (isTrusted ? trusted : shallow)++;
    if (!colored(frame, i)) {
      (isTrusted ? trustedEnd : shallowEnd)++;  // black for good: fine
    } else if (firstBlack[i]) {
      (isTrusted ? trustedFlash : shallowFlash)++;
      if (!isTrusted) shallowBlackFrames += blackFrames[i];
      flashAt[firstBlack[i]]++;
      if (dump && dumped < 30) {
        ++dumped;
        fprintf(stderr, "flash at (%zu, %zu): prior %d %d %d a%d, final %d %d %d, first black at pass %d, meta taken %d hits %d%s%s%s\n",
                i % r.w, i / r.w, prior[4 * i], prior[4 * i + 1], prior[4 * i + 2], a, frame[4 * i], frame[4 * i + 1],
                frame[4 * i + 2], firstBlack[i], samplesTaken(meta[i]), limitHits(meta[i]),
                isCertain(meta[i]) ? " certain" : "", isCoarseBlack(meta[i]) ? " coarse" : "",
                isBlack(meta[i]) ? " black" : "");
      }
    }
  }
  int worstPass = 0;
  for (int k = 1; k <= passesB; ++k)
    if (flashAt[k] > flashAt[worstPass]) worstPass = k;
  // Trusted colors must never flash. Boundary pixels (mixed color: the previous view's pixel
  // was part black) and the shallow surround fill are approximate: a pixel there whose own
  // sample hits is black until the sample escapes deeper, which is right by the rule that
  // black and color stay separate; they are reported with how long they stayed black.
  const double meanBlack = shallowFlash ? (double)shallowBlackFrames / shallowFlash : 0.0;
  const bool ok = trustedFlash == 0 && passesB > 1 && (factor >= 1 || shallow > 0);
  printf("flicker test (zoom x%g, %d passes before, %d after, surround %.3f ms): %zu pixels with trusted preview "
         "color (%zu black for good, %zu black in between), %zu with shallow or mixed color (%zu black for good, "
         "%zu black in between for %.1f frames on average, most first at pass %d) -> %s\n",
         factor, passesA, passesB, surroundMs, trusted, trustedEnd, trustedFlash, shallow, shallowEnd, shallowFlash,
         meanBlack, worstPass, ok ? "OK" : "FAIL");
  r.freeFrame();
  return ok ? 0 : 1;
}

// Self-test for halos in black areas: converges a view, zooms in by `factor` `steps` times
// quickly (one pass between zooms, as when the wheel is spun), then watches every frame of the
// final view until it has converged. The black and the non-black areas must stay separate: a
// pixel that is black by rule at the end (all samples hit, or blocked by the coarse image)
// must not have shown color in any frame before, which is what a magnified preview smearing
// boundary colors into the interior would do. A one-pixel band is tolerated: a pixel of the
// final view inside black may map to a boundary pixel of the earlier, coarser view whose
// mean was colored; beyond that band the halo is counted.
int testHalo(const Options& o, double factor, int steps) {
  Renderer r;
  r.w = o.width;
  r.h = o.height;
  r.view = o.view;
  r.autoIter = !o.iterGiven;
  r.maxIterCap = kFlickerTestCap;
  r.allocFrame();
  const size_t n = (size_t)r.w * r.h, bytes = n * 4;
  std::vector<uint8_t> frame(bytes);
  auto grab = [&]() {
    r.resolveDisplay();
    CUDA_CHECK(cudaMemcpy(frame.data(), r.display, bytes, cudaMemcpyDeviceToHost));
  };
  r.applyChanges();
  int passesA = 0;
  do {
    r.renderPasses(0);
    ++passesA;
  } while (!r.done());
  grab();
  // Quick zooms: one pass each, then the final view is watched from its preview frame on.
  std::vector<int> lastColored(n, -1);  // frame index at which the pixel last showed color (-1: never)
  std::vector<size_t> coloredAt;        // per frame: pixels colored that end up black (filled at the end)
  std::vector<std::vector<uint8_t>> history;  // frames of the final view (preview frame first)
  for (int s = 0; s < steps; ++s) {
    r.zoomAt(0.5 * r.w, 0.5 * r.h, factor);
    r.applyChanges();
    grab();
    if (s == steps - 1) history.push_back(frame);
    if (s < steps - 1) r.renderPasses(0);
  }
  int passesB = 0;
  do {
    r.renderPasses(0);
    ++passesB;
    grab();
    history.push_back(frame);
  } while (!r.done());
  std::vector<float> weights;
  std::vector<unsigned short> meta;
  readCounts(r.accum, r.w, r.h, weights, meta);
  // Final black by rule, and more than a pixel away from any final non-black pixel.
  std::vector<uint8_t> black(n, 0), deep(n, 0);
  size_t blackCount = 0, deepCount = 0;
  for (size_t i = 0; i < n; ++i) black[i] = showsBlack(meta[i]), blackCount += black[i];
  for (int y = 0; y < r.h; ++y)
    for (int x = 0; x < r.w; ++x) {
      const size_t i = (size_t)y * r.w + x;
      if (!black[i]) continue;
      bool inside = true;
      for (int dy = -1; dy <= 1 && inside; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= r.w || ny >= r.h) continue;
          if (!black[(size_t)ny * r.w + nx]) {
            inside = false;
            break;
          }
        }
      deep[i] = inside;
      deepCount += inside;
    }
  // Frame 0 is the preview itself (the previous image magnified, shown before any sample):
  // what it shows inside the final black is reported but not judged; the frames with samples
  // are.
  size_t halo = 0, haloBand = 0, worst = 0, inPreview = 0;
  int worstFrame = 0, lastFrame = -1;
  for (size_t f = 0; f < history.size(); ++f) {
    const std::vector<uint8_t>& img = history[f];
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
      if (!black[i] || (img[4 * i] | img[4 * i + 1] | img[4 * i + 2]) == 0) continue;
      if (f == 0) {
        inPreview += deep[i];
        continue;
      }
      if (deep[i]) {
        ++count;
        if (lastColored[i] < 0) ++halo;
        lastColored[i] = (int)f;
        lastFrame = (int)f;
      } else if (lastColored[i] < 0) {
        ++haloBand;
        lastColored[i] = (int)f;
      }
    }
    if (count > worst) worst = count, worstFrame = (int)f;
  }
  const bool ok = halo == 0 && passesB > 1 && deepCount > 0;
  printf("halo test (%d quick zooms of x%g at %dx%d, %d passes before, %d after): %zu pixels black at the end, %zu "
         "of them inside (not next to color); %zu of those colored in the preview frame; colored in a rendered "
         "frame: %zu inside (halo; worst frame %d with %zu, last in frame %d of %zu), %zu in the boundary band "
         "(tolerated) -> %s\n",
         steps, factor, r.w, r.h, passesA, passesB, blackCount, deepCount, inPreview, halo, worstFrame, worst,
         lastFrame, history.size(), haloBand, ok ? "OK" : "FAIL");
  r.freeFrame();
  return ok ? 0 : 1;
}

int runGui(const Options& o) {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  App app;
  app.win = SDL_CreateWindow("JyrkisFractals", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, o.width, o.height,
                             SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  if (!app.win) {
    fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
    return 1;
  }
  app.gl = SDL_GL_CreateContext(app.win);
  if (!app.gl) {
    fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_GL_MakeCurrent(app.win, app.gl);
  SDL_GL_SetSwapInterval(1);
  if (!loadGL()) {
    fprintf(stderr, "OpenGL buffer objects not available\n");
    return 1;
  }
  printf("OpenGL: %s / %s (video driver: %s)\n", glGetString(GL_RENDERER), glGetString(GL_VERSION),
         SDL_GetCurrentVideoDriver());

  // Prefer the CUDA device that is driving this GL context.
  unsigned count = 0;
  int devices[8];
  int dev = 0;
  if (cudaGLGetDevices(&count, devices, 8, cudaGLDeviceListAll) == cudaSuccess && count > 0) dev = devices[0];
  cudaGetLastError();
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("CUDA: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  app.view = o.view;
  app.homeView = ViewParams();
  app.homeView.palette = o.view.palette;
  app.autoIter = !o.iterGiven;
  app.createBuffers();
  if (!o.viewGiven) app.view.scale = defaultScale(app.w);
  printf("Interop: %s\n%s", app.interop ? "CUDA writes directly into the GL pixel buffer" : "host copy fallback",
         kHelp);

  auto start = std::chrono::steady_clock::now();
  bool running = true;
  SDL_Event e;
  while (running) {
    if (!app.busy() && SDL_WaitEventTimeout(&e, 100)) running = app.handleEvent(e);
    while (running && SDL_PollEvent(&e)) running = app.handleEvent(e);
    if (!running) break;
    app.applyChanges();
    if (app.recolored) {
      app.recolored = false;
      app.resolveToTexture();
      app.present();
      app.updateTitle();
    }
    if (app.previewPending && app.previewWorthShowing()) {
      // Show the resampled previous image right away; real samples start next iteration.
      app.resolveToTexture();
      app.present();
      app.updateTitle();
      continue;
    }
    if (!app.done()) {
      const auto t0 = std::chrono::steady_clock::now();
      app.renderPasses(kFrameBudgetMs);
      app.resolveToTexture();
      app.present();
      app.updateTitle();
      // Real time of this update, display included: a slow one makes the next passes cheaper.
      app.adaptBudget(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if (o.autoZoomSteps != 0) {
      static int stepsDone = 0;
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
      static const int stepMs = getenv("JF_ZOOM_MS") ? atoi(getenv("JF_ZOOM_MS")) : 400;
      if (stepsDone < std::abs(o.autoZoomSteps) && ms.count() >= stepMs * (stepsDone + 1)) {
        ++stepsDone;
        if (app.trace) {
          size_t freeMem = 0, totalMem = 0;
          cudaMemGetInfo(&freeMem, &totalMem);
          fprintf(stderr, "auto-zoom step %d (useful depth %d, verified %d, magnification %.2f, GPU memory in use %zu MB)\n",
                  stepsDone, app.displayUseful, app.displayVerified, app.displayMag,
                  (size_t)((totalMem - freeMem) >> 20));
        }
        app.zoomAt(0.5 * app.w, 0.5 * app.h, o.autoZoomSteps > 0 ? 1.25 : 0.8);
      }
    }
    if (o.exitAfterMs > 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
      if (ms.count() >= o.exitAfterMs) running = false;
    }
  }
  if (!o.saveOnExit.empty()) app.saveScreenshot(o.saveOnExit);
  printf("last view: center %.17g %+.17gi  scale %.17g  zoom %.4g  iter %d+%d  samples %d/%d  hits %d  "
         "useful depth %d  render %.2f ms\n",
         app.view.centerX, app.view.centerY, app.view.scale, app.zoom(), app.view.maxIter, app.iterAdjust,
         app.passes ? app.samplesActive() : 0, totalSamples(app.view), app.stats.hitCount, app.usefulIter,
         app.frameMs);

  app.destroyBuffers();
  SDL_GL_DeleteContext(app.gl);
  SDL_DestroyWindow(app.win);
  SDL_Quit();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (getenv("JF_NO_TILE_SKIP")) setTileSkip(false);
  Options o = parseArgs(argc, argv);
  if (o.dumpSamples) {
    std::vector<float2> pattern;
    samplePattern(o.view.samples, pattern);
    for (const float2& p : pattern) printf("%.5f %.5f\n", p.x, p.y);
    return 0;
  }
  if (!o.headlessOut.empty() || o.benchFrames > 0 || o.shiftTest || o.zoomTest || o.deeperTest || o.resumeTest ||
      o.recolorTest || o.flickerTest || o.haloTest)
    return runHeadless(o);
  return runGui(o);
}
