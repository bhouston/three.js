# Example assets on Land of Assets

The examples use the public `mrdoob/three` project with explicit `1.0.0` versions. `manifest.json` records all 113 repository uploads, their original sources and filenames, delivery URLs, and SHA256 hashes of the delivered files. Repository assets come from the upstream `dev` revision recorded in the manifest. Sponza is additionally installed from `KhronosGroup/glTF-Sample-Assets`. The WebGL and WebGPU glTF catalogs reuse the installed Damaged Helmet and Sponza models, and load 113 other available models directly from production Land of Assets URLs on selection. The 36 unmatched catalog models, plus other external Needle and MaterialX samples, retain their upstream URLs.

The uploads comprise 63 repository models, 4 IES profiles, 18 environment maps, 28 MaterialX packages. The 113 uploaded repository source assets and 37 orphaned buffers and textures have been removed from this branch. Shared textures, attribution, and license files remain in the repository.

## Install and run locally

From the repository root:

```sh
npm install
npx loa install --frozen-lockfile
npm start
```

`@landofassets/cli` is a dev dependency. `npm run assets:install` is an equivalent installer shortcut. No LoA account or login is required: the assets are public and the lockfile pins production `https://api.landofassets.com` references.

Commit `assets.json` and `assets-lock.json`, and leave generated `assets/` out of Git. Manifest keys determine the stable local filenames, for example `/assets/mrdoob/three/damagedhelmet.glb`. Versions stay in the manifest and lockfile, so changing a selection does not change the loader URL. The examples and manual snippets load ordinary local URLs with their existing loaders and no runtime adapter.

Run installation before serving examples or taking screenshots. Native bytes are verified and reused from the shared global cache; each installed native path links directly to its read-only global cache object, with independent copies as fallback when symlinks are unavailable. Aliases never point to another local asset. There is no additional project-local versioned store. Every `loa install`, `loa pull`, and `loa remove` prunes unreferenced local output while keeping cached objects intact. Copy `assets/` into deployments and dereference symlinks if your packaging tool does not follow them. Root-relative URLs assume the checkout is served at the origin root.

`manifest.json` retains the original hosted URLs as provenance and records each asset's `localUrl` for verification. The `externalAssets` section records the separately packaged Sponza asset and the remotely delivered Khronos GLBs. Only installed assets have a `localUrl`; catalog-only models are absent from `assets.json`.

## Khronos model catalogs

The WebGL and WebGPU glTF loader examples share `examples/models/gltf/model-index.json`, pinned to the Khronos source revision recorded in `manifest.json.catalog`. Damaged Helmet, the default selection, reuses `/assets/mrdoob/three/damagedhelmet.glb`; Sponza also uses its installed local URL. The other 113 matching Land of Assets uploads use direct versioned production URLs with `?original=true` and load only when selected. They are not installed by `loa install`. Existing uploaded GLBs are reused without conversion or re-upload. The 36 unmatched entries retain pinned upstream URLs, preserving the full selector; those entries still need internet access. The model index itself is served locally.

`manifest.json.externalAssets` records Sponza and the remotely delivered Khronos assets with their uploaded SHA256 hashes. Sponza is the only new upload in this update.

## Sponza

All three Sponza examples load `/assets/KhronosGroup/glTF-Sample-Assets/sponza.glb` through their existing loaders. The manifest pins `KhronosGroup/glTF-Sample-Assets/sponza` to `1.0.0`; no runtime model-index request or external texture downloads are needed.

The 52.6 MB GLB packages the [Khronos source at revision `f0920f5b32d172a30efe3f9257e965918831e6ea`](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/f0920f5b32d172a30efe3f9257e965918831e6ea/Models/Sponza). `gltf-transform copy Sponza.gltf sponza.glb` embeds the geometry buffer and all 69 images, without optimization or texture re-encoding. Verification compares all 405 decoded accessors, all image payloads, the node graph, primitive bindings, and material settings. Embedded asset metadata preserves source attribution and the original Crytek license link.

Sponza retains the [Cryengine Limited License Agreement](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/f0920f5b32d172a30efe3f9257e965918831e6ea/LICENSES/LicenseRef-CRYENGINE-Agreement.txt), credited to Crytek, Frank Meinl, Alexandre Pestana, and Morgan McGuire. The service share-license setting is `NONE`; no replacement license is assigned. Its source revision, package hash, and resource counts are recorded under `externalAssets` in `manifest.json`. There are 114 installed assets in total: 113 transferred repository assets and Sponza. Switching to a remote catalog model requires internet access.

## Packaging and delivery

- Land of Assets requires short asset names containing letters, numbers, and single dashes. Upload filenames retain their original names; asset names distinguish collisions and shorten long names.
- The service accepts GLB rather than loose glTF. The 16 repository `.gltf` documents are wrapped in GLB containers with their original buffers and images embedded as data URIs. Scene graphs, extensions, and resource bytes are preserved. Their uploaded filenames retain the original stem with a `.glb` extension.
- MaterialX documents are delivered as `.mtlx.zip` packages containing their resources. Textures are not resized or re-encoded. The CLI rejected the existing rotate3d validation sample, so that one package contains the original XML and texture zipped directly.
- Five UltraHDR JPEG environment maps are stored as `IMAGE` because the service restricts `ENVMAP` to HDR/EXR. Their bytes and example loaders are unchanged.
- Delivery URLs use `?original=true` to request the uploaded format. The four Needle models retain their original cloud URLs and progressive LOD hosting.
- Assets carry source attribution and use the service's `NONE` share-license setting to avoid assigning a new license. Original source licenses continue to apply.

## Verification

Anonymous downloads were checked against the manifest hashes with cross-origin delivery enabled. The 16 GLB containers preserve their source JSON graphs and all 42 embedded resources. Three.js parsed all four IES profiles, twelve HDR maps, and the EXR environment map. All MaterialX archive resources were checked for inclusion and byte preservation.

Core and addon unit tests pass with a hardware-backed Apple M3 GPU through ANGLE Metal. Screenshot comparison results are recorded in the pull request. No software rasterizer was used.
