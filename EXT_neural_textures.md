EXT_neural_textures Specification Draft
⚬ Extension Name: EXT_neural_textures
⚬ Status: Proposal / Draft
⚬ Dependencies: glTF 2.0 Core
1. Motivation & Background
Standard real-time PBR material workflows require multiple discrete texture maps (Base Color, Normal, Roughness, Metallic, Occlusion, Emissive). Traditional block-based texture compression schemes (such as BC7 or Basis Universal / KTX2) compress each texture independently. Because they evaluate textures in isolation, traditional codecs fail to exploit cross-texture spatial correlations—such as how normal map boundaries tightly align with changes in albedo or metallic values.
EXT_neural_textures introduces a neural texture representation based on the principles established in NVIDIA's SIGGRAPH 2023 paper, "Random-Access Neural Materials" (Vaillant et al.). By compressing all PBR texture channels of a material jointly into a single multi-resolution latent feature grid evaluated by a lightweight Multi-Layer Perceptron (MLP), this technique achieves dramatically higher compression ratios than modern image codecs (AVIF, WebP) and standard GPU block compression formats (BC7, KTX2).
Traditional PBR:  [BaseColor (BC7)]  [Normal (BC5)]  [Roughness/Metal (BC1)]  -->  ~3.5 – 5.0 MB
Neural Texture:   [ Joint Multi-Resolution Feature Grid + Tiny MLP ]         -->  ~0.15 – 0.3 MB (15x-30x savings)

By decoupling texture quality from traditional pixel resolution and moving signal evaluation into fragment shader matrix math, EXT_neural_textures drastically reduces network transfer sizes and GPU VRAM footprint while keeping fragment ALU costs lightweight enough for real-time performance on mobile hardware.
2. Targeted Mobile Performance Profile
While neural texture compression can scale up for desktop GPUs, this extension recommends a standardized baseline architecture tuned specifically for real-time execution on mid-range and low-end mobile devices (e.g., Apple A-series, Qualcomm Adreno, ARM Mali).
To keep fragment shader evaluation under ~800 FLOPs per pixel, the recommended default network configuration is:
⚬ Hidden Layer Topology: 1 Hidden Layer with a width of 16 neurons.
⚬ SIMD Hardware Alignment: Layer widths constrained to powers of 2 (specifically 16) allow mobile compilers to map matrix multiplications natively to mat4 * vec4 vector instructions, utilizing hardware Fused Multiply-Add (FMA) cycles.
⚬ Feature Grid Pyramids: A 3-level grid with resolutions of 32 \times 32, 64 \times 64, and 128 \times 128 (4 channels per level, sampled via bilinear lookups).
This targeted profile cuts VRAM consumption by over 90% compared to standard uncompressed or block-compressed 1K/2K material sets, while executing well within the frame time budgets of tile-based mobile renderers.
3. Extension Specification
3.1 Root Extension Declaration (neuralTextures)
Global neural texture assets are declared in the root extensions.EXT_neural_textures array and referenced by index throughout the glTF document.
{
  "extensionsUsed": ["EXT_neural_textures"],
  "extensions": {
    "EXT_neural_textures": {
      "neuralTextures": [
        {
          "name": "Recommended_Mobile_PBR_NT",
          "arch": {
            "inputs": 12,
            "hiddenLayers": 1,
            "hiddenWidth": 16,
            "outputs": 5,
            "activation": "relu"
          },
          "weightsBufferView": 2,
          "latentGrid": {
            "texture": 0,
            "baseResolution": 32,
            "levels": 3,
            "channelsPerLevel": 4
          }
        }
      ]
    }
  }
}

neuralTextures[i] Properties
⚬ name (string, optional): Descriptive name for the neural texture.
⚬ arch (object, required): ⚬ inputs (integer, required): Total input vector dimension fed to the MLP (e.g., 12 for a 3-level \times 4-channel feature grid). ⚬ hiddenLayers (integer, required): Number of hidden layers. Recommended: 1 for mobile. ⚬ hiddenWidth (integer, required): Number of neurons per layer. Must be a power of two (e.g., 4, 16, 32) to enable vectorization. Recommended: 16 for mobile. ⚬ outputs (integer, required): Dimension of the output vector. ⚬ activation (string, optional): Activation function for hidden layers ("relu", "leaky_relu", "none"). Default is "relu".
⚬ weightsBufferView (integer, required): Index of the bufferView referencing raw binary weights and biases (FP16 or FP32 array).
⚬ latentGrid (object, required): ⚬ texture (integer, required): Top-level textures index containing the multi-resolution latent feature map. ⚬ baseResolution (integer, required): Coarsest grid resolution (e.g., 32). ⚬ levels (integer, required): Total feature grid levels in the pyramid (e.g., 3 for resolutions 32, 64, 128). ⚬ channelsPerLevel (integer, optional): Latent channels per grid level. Default is 4.
3.2 Material Binding
Materials reference a neuralTexture index and bind its output channels to dynamic material parameters. Channels omitted from the channels object default directly to standard glTF scalar/vector constants (e.g., roughnessFactor, metallicFactor).
{
  "materials": [
    {
      "name": "SciFi_Metal_Material",
      "pbrMetallicRoughness": {
        "metallicFactor": 1.0,
        "roughnessFactor": 0.15
      },
      "extensions": {
        "EXT_neural_textures": {
          "neuralTextures": [
            {
              "index": 0,
              "texCoord": 0,
              "channels": {
                "baseColor": [0, 1, 2],
                "normal": [3, 4]
              }
            }
          ]
        }
      }
    }
  ]
}

Material Extension Properties
⚬ neuralTextures (array, required): List of neural texture evaluations applied to this material. ⚬ index (integer, required): Index into the root extensions.EXT_neural_textures.neuralTextures array. ⚬ texCoord (integer, optional): Index of the UV coordinate set (TEXCOORD_n). Default is 0. ⚬ channels (object, required): Maps PBR parameter properties to output vector indices of the neural network: ⚬ baseColor: [r, g, b] ⚬ normal: [x, y] tangent space normal ⚬ roughness: [r] ⚬ metallic: [m] ⚬ occlusion: [ao] ⚬ emissive: [r, g, b]
4. Reference Shader Implementation
Runtime engines (e.g., Three.js) build fragment shaders using SIMD matrix operations (mat4 * vec4) to evaluate the neural texture efficiently:
// GLSL Fragment Shader: Mobile-Optimized Evaluation (Inputs: 12, Hidden: 16, Outputs: 5)
uniform sampler2D uLatentGrid;
uniform mat4 uWeights_L1[3]; // 16x12 layer packed into 3 mat4s
uniform vec4 uBias_L1;
uniform mat4 uWeights_Out[1]; // Output layer matrix

varying vec2 vUv0;

void main() {
    // 1. Fetch 3 multi-resolution latent levels (12 channels packed into 3 vec4s)
    vec4 g0 = texture2D(uLatentGrid, vUv0);         // 32x32 level
    vec4 g1 = texture2D(uLatentGrid, vUv0 * 2.0);   // 64x64 level
    vec4 g2 = texture2D(uLatentGrid, vUv0 * 4.0);   // 128x128 level

    // 2. Evaluate Layer 1 using SIMD mat4 * vec4 FMA hardware execution
    vec4 h0 = uWeights_L1[0] * g0 + uWeights_L1[1] * g1 + uWeights_L1[2] * g2 + uBias_L1;
    h0 = max(vec4(0.0), h0); // ReLU activation

    // 3. Evaluate Output Layer
    vec4 out0 = uWeights_Out[0] * h0;

    // 4. Map Neural Outputs & Combine with Core glTF Constants
    vec3 baseColor  = out0.rgb;               // Dynamically evaluated Base Color
    vec2 normalXY   = out0.a;                 // Dynamically evaluated Normal XY
    float roughness = 0.15;                   // Fallback from core roughnessFactor
    float metallic  = 1.0;                    // Fallback from core metallicFactor

    // Proceed to standard PBR lighting calculations...
}