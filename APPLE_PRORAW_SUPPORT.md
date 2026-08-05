# Apple ProRAW Support Implementation

## Overview

This document describes the implementation of Apple ProRAW DNG support in the darktable fork, which enables darktable to import and process iPhone ProRAW files across different DNG versions and compression formats.

## Supported Apple ProRAW Variants

### ✅ DNG 1.3.0.0 (JPEG Compression)
- **Compression:** JPEG (TIFF compression code 6)
- **Photometric Interpretation:** Color Filter Array (CFA)
- **Status:** Already supported by upstream darktable
- **Characteristics:** Standard lossless JPEG compression with CFA layout

### ✅ DNG 1.6.0.0 (JPEG + Predictor Mode 7)
- **Compression:** JPEG (TIFF compression code 7) for SubIFD
- **Secondary IFD:** Lossy JPEG (0x884c) with Semantic Mask (52527)
- **Photometric Interpretation:** LinearRaw (34892) for SubIFD, Semantic Mask (52527) for SubIFD1
- **Status:** Fixed via LJpeg predictor mode 7 support
- **Characteristics:** Uses JPEG predictor mode 7 (not previously supported), includes ML segmentation mask that must be filtered out

### ✅ DNG 1.7.0.0 (JPEG XL Compression)
- **Compression:** JPEG XL (TIFF compression code 52546)
- **Photometric Interpretation:** LinearRaw (34892)
- **Status:** Fixed via JPEG XL decoder implementation
- **Characteristics:** Uses JPEG XL compression for main camera (48MP) on iPhone 16 Pro/17 Pro, 3-channel chunky RGB output

## Implementation Details

### Phase 1: IFD Selection Fix

**Problem:** DNG 1.6.0.0 files contain multiple IFDs, including a Semantic Mask IFD that was being mistakenly selected as the image data instead of the actual photo.

**Solution:** Modified `DngDecoder::dropUnsuportedChunks()` to filter out Semantic Mask IFDs based on photometric interpretation.

**Key Changes:**
- Added check for photometric interpretation 52527 (Semantic Mask)
- Reject IFDs with Semantic Mask photometric interpretation
- Allow other photometric interpretations to pass through (for previews, etc.)
- Raised log priority for "Multiple RAW chunks found" from EXTRA to WARNING

### Phase 2: JPEG XL (DNG 1.7) Support

**Problem:** DNG 1.7.0.0 files use JPEG XL compression (code 52546) which was not supported by rawspeed.

**Solution:** Implemented full JPEG XL decoder using libjxl reference implementation.

**Key Changes:**
- Added `JpegXlDecompressor.cpp/h` - JPEG XL decompressor implementation
- Added `JpegXlTileLayout.h` - Geometry and validation helpers for JPEG XL tiles
- Added `DngDeinterleave.h` - DNG 1.7 field de-interleaving support
- Added COLUMNINTERLEAVEFACTOR TIFF tag (0xCD43)
- Wired compression 52546 into DngDecoder and AbstractDngDecompressor
- Configured with WITH_JPEGXL CMake option (default ON)

### Phase 3: LJpeg Predictor Mode 7 Support

**Problem:** DNG 1.6.0.0 files use JPEG predictor mode 7, which was not supported (only mode 1 was supported).

**Solution:** Implemented support for all JPEG predictor modes 2-7 per ITU-T T.81 specification.

**Key Changes:**
- Extended predictor mode check from mode 1 only to modes 1-7
- Implemented `computePrediction()` helper for all seven predictor modes
- Added support for neighbor access (Ra, Rb, Rc) for modes 2-7
- Added restart interval support
- Added detection of mislabeled lossy JPEG tiles (Blackmagic CinemaDNG issue)
- Refactored LJpegDecoder to use DecodeSettings struct
- Added support for inverted tile geometry (DJI, Blackmagic)

## References and Inspiration

### Upstream PRs Used

1. **PR #971: Add JPEG XL (DNG 1.7 / Compression 52546) decompressor**
   - Repository: darktable-org/rawspeed
   - Author: MaykThewessen
   - Date: June 2026
   - Status: Open (unmerged as of mid-2026)
   - Purpose: JPEG XL decoder using libjxl, specifically tested with iPhone 16 Pro Max and iPhone 17 Pro ProRAW files
   - Chosen over PR #755 because it was more complete, CI-passing, and specifically tested with real ProRAW files

2. **PR #963: LJpeg: support predictor modes 2–7**
   - Repository: darktable-org/rawspeed
   - Author: da-phil
   - Date: April 2026
   - Status: Open (unmerged as of mid-2026)
   - Purpose: Support for JPEG predictor modes 2-7, needed for DJI, Blackmagic, and Apple ProRAW files
   - Addresses issues #189 and #258 in rawspeed

### Related Issues

- **rawspeed issue #516:** JPEG XL support request
- **rawspeed issue #258:** LJpegDecompressor predictor mode 7 support
- **darktable issue #18791:** DNG files are not supported (iPhone 16 Pro Max)

## Files Modified

### Build Configuration
- `CMakeLists.txt` - Added WITH_JPEGXL option
- `cmake/src-dependencies.cmake` - Added libjxl detection via pkg-config
- `src/config.h.in` - Added HAVE_JPEGXL define
- `.gitmodules` - Updated to point to personal fork (dtrtuser/rawspeed) with apple-prraw-support branch

### Decoder Files
- `src/librawspeed/decoders/DngDecoder.cpp` - Added photometric interpretation filtering, JPEG XL compression case, deinterleaveFields method
- `src/librawspeed/decoders/DngDecoder.h` - Added deinterleaveFields method declaration
- `src/librawspeed/decoders/CMakeLists.txt` - Added DngDeinterleave.h to sources
- `src/librawspeed/decoders/SimpleTiffDecoder.h` - Constructor initialization fix

### Decompressor Files
- `src/librawspeed/decompressors/AbstractDngDecompressor.cpp` - Added JPEG XL decompression thread, lossy JPEG detection, algorithm include
- `src/librawspeed/decompressors/CMakeLists.txt` - Added JPEG XL source files and libjxl linking
- `src/librawspeed/decompressors/JpegDecompressor.cpp` - Added data precision check
- `src/librawspeed/decompressors/LJpegDecoder.cpp` - Added predictor modes 2-7 support, restart interval support, inverted tile geometry
- `src/librawspeed/decompressors/LJpegDecoder.h` - Added DecodeSettings struct, predictor mode support

### New Files Created
- `src/librawspeed/decoders/DngDeinterleave.h` - DNG 1.7 field de-interleaving implementation
- `src/librawspeed/decompressors/JpegXlDecompressor.cpp` - JPEG XL decompressor implementation
- `src/librawspeed/decompressors/JpegXlDecompressor.h` - JPEG XL decompressor header
- `src/librawspeed/decompressors/JpegXlTileLayout.h` - JPEG XL tile geometry and validation helpers

### TIFF Tags
- `src/librawspeed/tiff/TiffTag.h` - Added COLUMNINTERLEAVEFACTOR tag (0xCD43)

### IO Files
- `src/librawspeed/io/FileReader.cpp` - Minor fix (from PR #963)

### Fuzz Test Files
- `fuzz/librawspeed/decompressors/LJpegDecompressor.cpp` - Updated for new DecodeSettings struct (from PR #963)

## Build Dependencies

### Windows/MSYS2 UCRT64
- **Package:** `mingw-w64-ucrt-x86_64-libjxl` (version 0.12.0)
- **Installation:** `pacman -S mingw-w64-ucrt-x86_64-libjxl`
- **Status:** Already included in official darktable Windows build instructions

### Fedora
- **Package:** `libjxl-devel` (via dnf builddep)
- **Installation:** `sudo dnf install libjxl-devel` or via `dnf builddep darktable`
- **Status:** Available through standard package manager

## Repository Structure

### Submodule Configuration
- **Parent Repository:** https://github.com/dtrtuser/darktable (branch: filminversion)
- **Submodule Repository:** https://github.com/dtrtuser/rawspeed (branch: apple-prraw-support)
- **Submodule Path:** src/external/rawspeed
- **Branch Configuration:** Set to `apple-prraw-support` in `.gitmodules`

### Commit History
1. `8b440ca7` - Add JPEG XL (DNG 1.7) support for Apple ProRAW
2. `4b640882` - Add LJpeg predictor modes 2-7 support for Apple ProRAW
3. `7cf3dc3b` - Tune AGENTS.md (base commit)
4. `28f97e4f` - Merge branch 'darktable-org:develop' into apple-prraw-support (upstream sync)

## Testing

### Test Files
The implementation was tested with three different Apple ProRAW variants:
- **Sample A:** DNG 1.6.0.0 with JPEG compression and predictor mode 7
- **Sample B:** DNG 1.7.0.0 with JPEG XL compression (iPhone 16 Pro Max, iPhone 17 Pro)
- **Sample C:** DNG 1.3.0.0 with standard JPEG compression

### Test Results
- ✅ All three variants import successfully
- ✅ No "file has unknown format" errors
- ✅ No "Unsupported predictor mode" errors
- ✅ Semantic Mask IFDs correctly filtered out
- ✅ JPEG XL tiles decode correctly with proper dimensions
- ✅ LinearRaw photometric interpretation handled correctly

## AI Usage Disclosure

This implementation was created with the assistance of Devin (AI coding assistant). All code changes were reviewed, tested, and committed by a conscious human developer. The AI tool was used for:
- Code generation based on upstream PR specifications
- Build system configuration
- Git repository management
- Documentation generation

The human developer bears full responsibility for all contributions and understands the technical implementation and its implications.

## Future Maintenance

### Upstream Sync Workflow
To sync with upstream darktable-org/rawspeed develop branch:
```bash
cd src/external/rawspeed
git fetch origin
git merge origin/develop
git pull --rebase fork apple-prraw-support
git push fork apple-prraw-support
cd ../..
git add src/external/rawspeed
git commit -m "Update rawspeed submodule with upstream changes"
git push origin filminversion
```

### Potential Conflicts
- Both PRs (#963 and #971) are still open in upstream, so future upstream merges may include similar changes
- If upstream merges these PRs, the custom implementation may need to be removed/replaced
- The personal fork approach allows for independent maintenance regardless of upstream decisions

## Conclusion

This implementation successfully adds comprehensive Apple ProRAW support to the darktable fork by integrating two well-tested upstream PRs. All three major ProRAW variants (DNG 1.3, 1.6, and 1.7) are now supported, enabling users to import and process iPhone ProRAW files in darktable.
