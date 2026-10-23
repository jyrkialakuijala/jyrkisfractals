#include "image_io.h"

#include <zlib.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void putBE32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)(v >> 16);
  p[2] = (uint8_t)(v >> 8);
  p[3] = (uint8_t)v;
}

bool writeChunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
  uint8_t hdr[8];
  putBE32(hdr, (uint32_t)len);
  memcpy(hdr + 4, type, 4);
  uint32_t crc = crc32(0, hdr + 4, 4);
  if (len) crc = crc32(crc, data, (uInt)len);
  uint8_t crcb[4];
  putBE32(crcb, crc);
  return fwrite(hdr, 1, 8, f) == 8 && (len == 0 || fwrite(data, 1, len, f) == len) && fwrite(crcb, 1, 4, f) == 4;
}

}  // namespace

bool writePNG(const char* path, const uint8_t* rgba, int w, int h) {
  const size_t rowBytes = (size_t)w * 3 + 1;
  std::vector<uint8_t> raw(rowBytes * h);
  for (int y = 0; y < h; ++y) {
    uint8_t* row = raw.data() + rowBytes * y;
    row[0] = 0;  // filter: none
    const uint8_t* src = rgba + (size_t)y * w * 4;
    for (int x = 0; x < w; ++x) {
      row[1 + x * 3 + 0] = src[x * 4 + 0];
      row[1 + x * 3 + 1] = src[x * 4 + 1];
      row[1 + x * 3 + 2] = src[x * 4 + 2];
    }
  }
  uLongf destLen = compressBound((uLong)raw.size());
  std::vector<uint8_t> z(destLen);
  if (compress2(z.data(), &destLen, raw.data(), (uLong)raw.size(), 6) != Z_OK) return false;

  FILE* f = fopen(path, "wb");
  if (!f) return false;
  static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  uint8_t ihdr[13];
  putBE32(ihdr, (uint32_t)w);
  putBE32(ihdr + 4, (uint32_t)h);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 2;   // color type: RGB
  ihdr[10] = 0;  // compression
  ihdr[11] = 0;  // filter
  ihdr[12] = 0;  // interlace
  bool ok = fwrite(sig, 1, 8, f) == 8 && writeChunk(f, "IHDR", ihdr, 13) &&
            writeChunk(f, "IDAT", z.data(), destLen) && writeChunk(f, "IEND", nullptr, 0);
  ok = (fclose(f) == 0) && ok;
  return ok;
}
