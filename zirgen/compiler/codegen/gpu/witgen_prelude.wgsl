// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================================
// WGSL witgen prelude for the zirgen WGSL backend.
//
// The zirgen WGSL backend emits `steps.wgsl` (the witness-generation step
// functions) referencing the field types, field arithmetic, and witness-buffer
// access helpers defined here. WGSL has no #include, so the consumed shader
// module is the concatenation:
//
//     witgen_prelude.wgsl  +  types.wgsl.inc  +  layout.wgsl.inc  +  steps.wgsl
//
// plus a circuit-specific `@compute` entry point (risc0-side) that sets `cycle`
// from the dispatch id and calls `step_Top` / `step_TopAccum`.
//
// This prelude mirrors `risc0/circuit/rv32im/src/prove/hal/rust_steps.rs` (the
// CPU witgen reference) and `risc0_core/src/field/baby_bear.rs` (the field).
// The field-arithmetic helpers (add/sub/mul/mul_wide/ext_*) are the verified
// implementations from `risc0/zkp/src/hal/webgpu_codegen/prelude.wgsl`.
// ============================================================================

// ----- BabyBear field constants ---------------------------------------------
const P: u32 = 2013265921u;       // 15 * 2^27 + 1
const M: u32 = 2281701377u;       // -P^-1 mod 2^32  (0x88000001)
const R2: u32 = 1172168163u;      // R^2 mod P,  R = 2^32
const NBETA: u32 = 1073741848u;   // extension-field BETA constant
const MONT_ONE: u32 = 268435454u; // encode(1) = R mod P
const INVALID: u32 = 0xffffffffu; // unset-cell sentinel

// ----- Field types ----------------------------------------------------------
// WGSL has no generics; the zirgen WGSL backend lowers field elements to plain
// u32 (Montgomery form) and the degree-4 extension to vec4<u32>.
alias Val = u32;
alias ExtVal = vec4<u32>;
alias Index = u32;

// A column reference. The backend lowers `RefAttr` to `Reg(Nu)`.
struct Reg {
  col: u32,
}
// The leaf monomorphized BoundLayout wrapper. The per-circuit composite
// BoundLayout_<T> structs are emitted into types.wgsl.inc by emitLayoutDef;
// BoundLayout_Reg is the universal leaf and lives here in the prelude.
// The layout field is named `lyt` because `layout` is a WGSL reserved keyword.
struct BoundLayout_Reg {
  lyt: Reg,
  buf: u32,
}

// ----- Witness buffers ------------------------------------------------------
// Buffer ids (the backend lowers `get_buffer(name)` to `buf_<name>`, and the
// @compute entry passes these into step_Top / step_TopAccum).
const buf_data: u32 = 0u;
const buf_global: u32 = 1u;
const buf_accum: u32 = 2u;
const buf_mix: u32 = 3u;

// `cycle` is the witness row currently being generated. The @compute entry
// point sets it from @builtin(global_invocation_id) before calling a step fn.
var<private> cycle: u32;

struct WitgenParams {
  data_rows: u32,
  global_rows: u32,
  accum_rows: u32,
  mix_rows: u32,
  // accum's BufferRow::with_zero_back_after column (0 = no zero-back rule).
  accum_zero_back: u32,
}

@group(0) @binding(0) var<storage, read_write> data_buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> global_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> accum_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> mix_buf: array<u32>;
@group(0) @binding(4) var<uniform> params: WitgenParams;

// ----- BabyBear scalar arithmetic (Montgomery form) -------------------------
// Verbatim from risc0/zkp/src/hal/webgpu_codegen/prelude.wgsl, which is
// verified to match baby_bear.rs.

fn add(lhs: Val, rhs: Val) -> Val {
  let sum = lhs + rhs;
  if (sum >= P) {
    return sum - P;
  }
  return sum;
}

fn sub(lhs: Val, rhs: Val) -> Val {
  if (lhs >= rhs) {
    return lhs - rhs;
  }
  return lhs + P - rhs;
}

fn mul_wide(lhs: u32, rhs: u32) -> vec2<u32> {
  let lhs_lo = lhs & 0xffffu;
  let lhs_hi = lhs >> 16u;
  let rhs_lo = rhs & 0xffffu;
  let rhs_hi = rhs >> 16u;
  let p0 = lhs_lo * rhs_lo;
  let p1 = lhs_hi * rhs_lo;
  let p2 = lhs_lo * rhs_hi;
  let p3 = lhs_hi * rhs_hi;
  let carry = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
  let lo = (p0 & 0xffffu) | ((carry & 0xffffu) << 16u);
  let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (carry >> 16u);
  return vec2<u32>(lo, hi);
}

fn mul(lhs: Val, rhs: Val) -> Val {
  let product = mul_wide(lhs, rhs);
  let low = 0u - product.x;
  let red = M * low;
  let red_product = mul_wide(red, P);
  var ret = product.y + red_product.y;
  if (product.x + red_product.x < product.x) {
    ret = ret + 1u;
  }
  if (ret >= P) {
    return ret - P;
  }
  return ret;
}

// Montgomery encode (direct -> Montgomery) and decode (Montgomery -> direct).
fn encode(a: u32) -> Val {
  return mul(R2, a);
}
fn decode(a: Val) -> u32 {
  return mul(1u, a);
}

// Square-and-multiply exponentiation in the Montgomery domain.
fn pow(base: Val, exp: u32) -> Val {
  var result: Val = MONT_ONE;
  var b: Val = base;
  var e: u32 = exp;
  while (e != 0u) {
    if ((e & 1u) != 0u) {
      result = mul(result, b);
    }
    b = mul(b, b);
    e = e >> 1u;
  }
  return result;
}

// Multiplicative inverse via Fermat's little theorem: x^(P-2). inv(0) = 0,
// matching baby_bear.rs.
fn inv(x: Val) -> Val {
  return pow(x, P - 2u);
}

// ----- BabyBearExt arithmetic (vec4<u32>, Montgomery components) -------------

fn ext_add(lhs: ExtVal, rhs: ExtVal) -> ExtVal {
  return ExtVal(add(lhs.x, rhs.x), add(lhs.y, rhs.y), add(lhs.z, rhs.z), add(lhs.w, rhs.w));
}

fn ext_sub(lhs: ExtVal, rhs: ExtVal) -> ExtVal {
  return ExtVal(sub(lhs.x, rhs.x), sub(lhs.y, rhs.y), sub(lhs.z, rhs.z), sub(lhs.w, rhs.w));
}

fn ext_mul(lhs: ExtVal, rhs: ExtVal) -> ExtVal {
  return ExtVal(
    add(mul(lhs.x, rhs.x),
        mul(NBETA, add(add(mul(lhs.y, rhs.w), mul(lhs.z, rhs.z)), mul(lhs.w, rhs.y)))),
    add(add(mul(lhs.x, rhs.y), mul(lhs.y, rhs.x)),
        mul(NBETA, add(mul(lhs.z, rhs.w), mul(lhs.w, rhs.z)))),
    add(add(add(mul(lhs.x, rhs.z), mul(lhs.y, rhs.y)), mul(lhs.z, rhs.x)),
        mul(NBETA, mul(lhs.w, rhs.w))),
    add(add(add(mul(lhs.x, rhs.w), mul(lhs.y, rhs.z)), mul(lhs.z, rhs.y)), mul(lhs.w, rhs.x)),
  );
}

fn ext_scale(lhs: ExtVal, rhs: Val) -> ExtVal {
  return ExtVal(mul(lhs.x, rhs), mul(lhs.y, rhs), mul(lhs.z, rhs), mul(lhs.w, rhs));
}

// TODO(wgsl): the extension-field inverse. The base-field path (inv/inv_0) is
// exact; ext_inv is a placeholder until verified against ExtElem::inv. iter 6's
// byte-identical SP-CR check will flag any rv32im path that actually needs it.
fn ext_inv(x: ExtVal) -> ExtVal {
  return x;
}

// ----- DSL builtin field helpers --------------------------------------------
// These names are emitted verbatim by the witgen step functions; they mirror
// the helpers in risc0's CUDA witgen.h / rust_steps.rs.

fn isz(x: Val) -> Val {
  if (x == 0u) {
    return MONT_ONE;
  }
  return 0u;
}

fn neg_0(x: Val) -> Val {
  return sub(0u, x);
}

fn inv_0(x: Val) -> Val {
  return inv(x);
}

// `mod` is a WGSL reserved keyword, so canonIdent escapes the DSL builtin to
// `mod_`; the prelude definition matches.
fn mod_(lhs: Val, rhs: Val) -> Val {
  return encode(decode(lhs) % decode(rhs));
}

// bitAnd / inRange keep the camelCase DSL builtin names (canonIdent leaves
// already-camelCase identifiers unchanged).
fn bitAnd(lhs: Val, rhs: Val) -> Val {
  return encode(decode(lhs) & decode(rhs));
}

fn inRange(low: Val, mid: Val, high: Val) -> Val {
  let l = decode(low);
  let m = decode(mid);
  let h = decode(high);
  if (l <= m && m < h) {
    return MONT_ONE;
  }
  return 0u;
}

// Montgomery Val -> plain u32, used for array/index conversions.
fn to_size_t(v: Val) -> u32 {
  return decode(v);
}

// eqz is a witness consistency assertion (it never writes witness state). For
// now it is a no-op so the witness is still produced.
// TODO(wgsl): route a failure to a debug error-flag buffer instead of dropping.
fn eqz(v: Val) {
}
fn eqz_ext(v: ExtVal) {
}

// ----- Witness buffer access (column-major; mirrors rust_steps.rs BufferRow) -
// A buffer cell (row, col) lives at `col * rows + row`. `data` and `accum` are
// mutable per-cycle buffers; `global` and `mix` are global (single-row).

fn buf_rows(buf_id: u32) -> u32 {
  switch buf_id {
    case 0u: { return params.data_rows; }
    case 1u: { return params.global_rows; }
    case 2u: { return params.accum_rows; }
    default: { return params.mix_rows; }
  }
}

fn buf_is_global(buf_id: u32) -> bool {
  return buf_id == buf_global || buf_id == buf_mix;
}

fn buf_get(buf_id: u32, idx: u32) -> u32 {
  switch buf_id {
    case 0u: { return data_buf[idx]; }
    case 1u: { return global_buf[idx]; }
    case 2u: { return accum_buf[idx]; }
    default: { return mix_buf[idx]; }
  }
}

fn buf_set(buf_id: u32, idx: u32, v: u32) {
  switch buf_id {
    case 0u: { data_buf[idx] = v; }
    case 1u: { global_buf[idx] = v; }
    case 2u: { accum_buf[idx] = v; }
    default: { mix_buf[idx] = v; }
  }
}

fn load(col: u32, buf_id: u32, back: u32) -> Val {
  // accum's zero-back rule (BufferRow::with_zero_back_after).
  if (buf_id == buf_accum && params.accum_zero_back != 0u
      && col > params.accum_zero_back && back > 0u) {
    return 0u;
  }
  let rows = buf_rows(buf_id);
  var row: u32;
  if (buf_is_global(buf_id)) {
    row = 0u;
  } else {
    row = (rows + cycle - back) % rows;
  }
  // TODO(wgsl): mirror BufferRow's is_valid()/valid_or_zero() checked-read
  // behavior; for now an unset (INVALID) cell is returned as-is.
  return buf_get(buf_id, col * rows + row);
}

fn load_ext(col: u32, buf_id: u32, back: u32) -> ExtVal {
  return ExtVal(load(col, buf_id, back),
                load(col + 1u, buf_id, back),
                load(col + 2u, buf_id, back),
                load(col + 3u, buf_id, back));
}

// Base-field value promoted to the canonical (x, 0, 0, 0) ExtVal embedding,
// matching F::ExtElem::from_subfield(F::Elem).
fn load_as_ext(col: u32, buf_id: u32, back: u32) -> ExtVal {
  return ExtVal(load(col, buf_id, back), 0u, 0u, 0u);
}

fn store(col: u32, buf_id: u32, v: Val) {
  let rows = buf_rows(buf_id);
  var row: u32;
  if (buf_is_global(buf_id)) {
    row = 0u;
  } else {
    row = cycle;
  }
  buf_set(buf_id, col * rows + row, v);
}

fn store_ext(col: u32, buf_id: u32, v: ExtVal) {
  store(col, buf_id, v.x);
  store(col + 1u, buf_id, v.y);
  store(col + 2u, buf_id, v.z);
  store(col + 3u, buf_id, v.w);
}

// ----- Externs --------------------------------------------------------------
// `invoke_extern` lowers here. assert/log/print are no-ops in witgen (matching
// CUDA witgen.h) and collapse to extern_noop(). The remaining externs read from
// the preflight trace -- in CUDA, ExecContext::preflight. The bodies below are
// TODO(wgsl) stubs with the correct signatures and zeroed results; uploading
// the PreflightTrace as GPU buffers and implementing the reads is a later
// iteration (the dispatch/buffer wiring is risc0-side). Array-returning
// externs return a WGSL array<Val,N>, which emitSaveResults projects per result.

fn extern_noop() {
}

fn extern_lookupDelta(table: Val, index: Val, count: Val) {
}

fn extern_lookupCurrent(table: Val, index: Val) -> Val {
  return 0u;
}

fn extern_memoryDelta(addr: Val, txn_cycle: Val, data_low: Val, data_high: Val, count: Val) {
}

fn extern_getDiffCount(txn_cycle: Val) -> Val {
  return 0u;
}

fn extern_isFirstCycle_0() -> Val {
  return 0u;
}

fn extern_hostReadPrepare(fp: Val, len: Val) -> Val {
  return 0u;
}

fn extern_hostWrite(fd: Val, addr_low: Val, addr_high: Val, len: Val) -> Val {
  return 0u;
}

fn extern_getMemoryTxn(addr: Val) -> array<Val, 5> {
  return array<Val, 5>(0u, 0u, 0u, 0u, 0u);
}

fn extern_divide(numer_low: Val, numer_high: Val, denom_low: Val, denom_high: Val,
                 sign_type: Val) -> array<Val, 4> {
  return array<Val, 4>(0u, 0u, 0u, 0u);
}

fn extern_getMajorMinor() -> array<Val, 2> {
  return array<Val, 2>(0u, 0u);
}

fn extern_nextPagingIdx() -> array<Val, 2> {
  return array<Val, 2>(0u, 0u);
}

fn extern_bigIntExtern() -> array<Val, 16> {
  return array<Val, 16>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}
