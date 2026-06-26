# Arm Neon (Advanced SIMD) Support Matrix: AWS, GCP, & Azure

Every 64-bit Arm (AArch64) instance deployed across AWS, Google Cloud, and Microsoft Azure natively supports the Arm Neon instruction set. Neon is a mandatory baseline requirement of the ARMv8-A and ARMv9-A architectures, meaning any Arm-based virtual machine you provision will automatically include Neon hardware vectorization capabilities.

---

## 1. Cloud Infrastructure Overview

### Amazon Web Services (AWS)
AWS utilizes its custom-designed **Graviton** hardware processor family:
* **Graviton4 & Graviton5:** Built on Arm Neoverse V2 cores. Features dual 128-bit Neon vector pipelines along with SVE2 support. (Families: `R8g`, `C8g`, `M8g`)
* **Graviton3:** Built on Arm Neoverse V1 cores. Includes dual 128-bit Neon pipelines and 256-bit SVE engines. (Families: `C7g`, `M7g`, `R7g`)
* **Graviton2:** Built on Arm Neoverse N1 cores. Contains dual 128-bit Neon execution units. (Families: `T4g`, `M6g`, `C6g`, `R6g`)

### Google Cloud Platform (GCP)
Google offers Arm architecture using a mix of custom Google silicon and partner hardware:
* **Google Axion:** Custom silicon built on Arm Neoverse V2 cores. Features high-throughput dual 128-bit Neon engines and SVE2. (Families: `C4A`, `X4A`)
* **Ampere Altra:** Merchant silicon built on Arm Neoverse N1 cores. Houses dual 128-bit Neon SIMD units per core. (Families: Tau `T2A`)

### Microsoft Azure
Azure utilizes specialized high-density custom chips and standard cloud processors:
* **Azure Cobalt 200:** Custom silicon built on the Arm Neoverse V3 platform. Features enhanced dual 128-bit Neon units and SVE2. (Latest Preview Instances)
* **Azure Cobalt 100:** Custom silicon built on Arm Neoverse N2 design. Includes dual 128-bit Neon vector processors. (Families: `Dpsv6`, `Dplsv6`, `Epsv6`)
* **Ampere Altra / Altra Max:** Enterprise silicon built on Arm Neoverse N1 cores. Features dual 128-bit Neon SIMD units per core. (Families: `Dpsv5`, `Dplsv5`, `Epsv5`)

---

## 2. Multi-Cloud Hardware Comparison Matrix

| Cloud Provider | Chip Model | Arm Core Baseline | Neon Register Width | Additional Extensions | Example VM / Instance Series |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AWS** | Graviton5 | Neoverse V2 | 2x 128-bit | SVE2, Crypto, Matrix Multiply | `R8g`, `C8g`, `M8g` |
| **AWS** | Graviton4 | Neoverse V2 | 2x 128-bit | SVE2, Crypto, BF16, INT8 | `R8g`, `C8g`, `M8g` |
| **AWS** | Graviton3 | Neoverse V1 | 2x 128-bit | SVE (2x 256-bit), BF16 | `C7g`, `M7g`, `R7g` |
| **AWS** | Graviton2 | Neoverse N1 | 2x 128-bit | Crypto, INT8 Dot Product | `T4g`, `M6g`, `C6g`, `R6g` |
| **GCP** | Google Axion | Neoverse V2 | 2x 128-bit | SVE2, Crypto, BF16, INT8 | `C4A`, `X4A` |
| **GCP** | Ampere Altra | Neoverse N1 | 2x 128-bit | Crypto, INT8 Dot Product | Tau `T2A` |
| **Azure** | Cobalt 200 | Neoverse V3 | 2x 128-bit | SVE2, Crypto, SME, MatMul | Cobalt 200 Series Preview |
| **Azure** | Cobalt 100 | Neoverse N2 | 2x 128-bit | SVE2, Crypto, BF16, MatMul | `Dpsv6`, `Dplsv6`, `Epsv6` |
| **Azure** | Ampere Altra | Neoverse N1 | 2x 128-bit | Crypto, INT8 Dot Product | `Dpsv5`, `Dplsv5`, `Epsv5` |

---

## 3. Production Compiler Flag Configurations

To fully exploit Neon and auto-vectorization across GCC and Clang, use the specialized `-mcpu` hardware optimization flags listed below:

### Target: Neoverse V2 / V3 (Graviton4/5, Google Axion, Cobalt 200)
```bash
# Recommended for maximum optimization on modern ARMv9-A instances
-O3 -mcpu=neoverse-v2
```

### Target: Neoverse N2 (Azure Cobalt 100)
```bash
# For ARMv9-A N2 profiles enabling both Neon and baseline SVE2
-O3 -mcpu=neoverse-n2
```

### Target: Neoverse V1 (Graviton3)
```bash
# Leverages dual 256-bit wide SVE alongside standard Neon SIMD blocks
-O3 -mcpu=neoverse-v1
```

### Target: Neoverse N1 (Graviton2, Ampere Altra VMs)
```bash
# Universal ARMv8.2-A target profile covering older generation VMs
-O3 -mcpu=neoverse-n1
```

### Generic/Portable Multi-Cloud Binary
```bash
# Safe fallback targeting all 64-bit Arm architectures natively with Neon
-O3 -march=armv8-a+simd
```
