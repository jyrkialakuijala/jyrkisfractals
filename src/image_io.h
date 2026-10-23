#pragma once
#include <cstdint>

// Writes an 8-bit RGB PNG (alpha is dropped). Returns true on success.
bool writePNG(const char* path, const uint8_t* rgba, int w, int h);
