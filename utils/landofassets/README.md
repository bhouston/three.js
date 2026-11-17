# Example assets on Land of Assets

The examples use the public `BenHouston3D/three` project with explicit `1.0.0` versions. `manifest.json` records all 113 repository uploads, their original sources and filenames, delivery URLs, and SHA256 hashes of the delivered files. Repository assets come from the upstream `dev` revision recorded in the manifest. Externally hosted Khronos, Needle, and MaterialX samples continue to use their upstream URLs.

The uploads comprise 63 repository models, 4 IES profiles, 18 environment maps, 28 MaterialX packages. The 113 uploaded repository source assets and 37 orphaned buffers and textures have been removed from this branch. Shared textures, attribution, and license files remain in the repository.

## Install and run locally

From the repository root:

```sh
npm install
npx loa install --frozen-lockfile
npm start
```

`@landofassets/cli` is a dev dependency. `npm run assets:install` is an equivalent installer shortcut. No LoA account or login is required: the assets are public and the lockfile pins production `https://api.landofassets.com` references.

Commit `assets.json` and `assets-lock.json`, and leave generated `assets/` out of Git. Manifest keys determine the stable local filenames, for example `/assets/BenHouston3D/three/damagedhelmet.glb`. Versions stay in the manifest and lockfile, so changing a selection does not change the loader URL. The examples and manual snippets load ordinary local URLs with their existing loaders and no runtime adapter.

Run installation before serving examples or taking screenshots. Native bytes are verified and reused from the shared global cache; installed project files are independent clones or copies. Alias bindings selecting the same version use relative symlinks when supported, with copies as fallback. There is no additional project-local versioned store. Copy `assets/` into deployments and dereference symlinks if your packaging tool does not follow them. Root-relative URLs assume the checkout is served at the origin root.

`manifest.json` retains the original hosted URLs as provenance and records each asset's `localUrl` for verification. External Khronos, Needle, and MaterialX URLs remain external.

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
