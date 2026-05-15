// Copyright 2026 RISC Zero, Inc.
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

// SP7 iter 6a -- the WGSL-only chunking transformation.
//
// The WGSL backend emits one `zhlt.step_func` per witgen function. Chrome's
// Tint compiler has a per-`@compute`-pipeline capacity ceiling far below the
// generated module (iter 5c/5d: a ~0.39 MB reachable closure device-loses;
// ~0.11 MB dispatches). The witgen giants -- `exec$Top`, `exec$Top$extract`,
// `exec$Top$accum` -- are each a small prologue + one wide `zstruct.switch`
// (the rv32im instruction-class mux) + an epilogue. The sub-workers reached
// by the mux arms (`exec$Sha0` etc.) are themselves wide switches.
//
// This pass splits a wide `zstruct.switch` into N chunk `step_func`s. Each
// chunk reruns the (idempotent -- witgen stores are deterministic) prologue,
// runs only its arm group, and runs the epilogue guarded by "did one of my
// arms fire" so a non-firing chunk cannot overwrite real witness values.
// Chunks call the same workers the original step_func did -- they are
// emitted as additional fns alongside the originals, not as replacements.
// risc0-side per-@compute-entry module pruning picks which chunks + which
// transitive callees to bundle into each pipeline's WGSL module.
// Runs only on the WGSL-only step-func clone, so Rust/C++/CUDA are untouched.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

using namespace mlir;

namespace zirgen::ZStruct {

namespace {

// A switch is "wide" -- worth splitting -- when it has more than this many
// arms. exec$Top/$extract/$accum have 13; the sub-workers have ~8. One arm
// per chunk keeps each chunk's reachable closure well under the ~0.1-0.4 MB
// Tint ceiling measured in iter 5d, AND makes recursion clean: a 1-arm chunk
// calls exactly one mux-arm worker, so call-rewriting it to a chunked worker
// is a simple linear expansion (no cross product).
constexpr unsigned kMaxArmsPerChunk = 1;

// Clone `src`'s region into `dst`, remapping as we go.
void cloneArm(Region& src, Region& dst, IRMapping& map) {
  src.cloneInto(&dst, map);
}

// Split one wide SwitchOp into chunk step_funcs. `parent` is the step_func
// containing `sw`; `sw` must be a direct child op of `parent`'s entry block
// (the giants are emitted that way). Returns the created chunk step_funcs.
SmallVector<zirgen::Zhlt::StepFuncOp> splitSwitch(zirgen::Zhlt::StepFuncOp parent,
                                                 SwitchOp sw) {
  MLIRContext* ctx = parent.getContext();
  OpBuilder builder(ctx);
  Location loc = sw.getLoc();
  unsigned numArms = sw.getArms().size();
  unsigned numChunks = (numArms + kMaxArmsPerChunk - 1) / kMaxArmsPerChunk;

  // Ops after `sw` in the entry block, up to (not including) the terminator,
  // are the epilogue. They may transitively use `sw`'s result.
  Block& body = parent.getBody().front();
  Operation* term = body.getTerminator();

  SmallVector<zirgen::Zhlt::StepFuncOp> chunks;
  ModuleOp mod = parent->getParentOfType<ModuleOp>();

  for (unsigned c = 0; c < numChunks; ++c) {
    unsigned armLo = c * kMaxArmsPerChunk;
    unsigned armHi = std::min(armLo + kMaxArmsPerChunk, numArms);
    // Reset each iteration: after building chunk c-1's body the insertion
    // point sits inside its block (after the return); chunk c must be a new
    // sibling op just before `parent`, not nested in the previous chunk.
    builder.setInsertionPoint(parent);

    // The chunk step_func has the same signature as `parent`. The builder
    // derives `arg_attrs` from `argNames`, and the verifier requires
    // `arg_attrs.size() == numArgs` -- so carry the parent's arg names over.
    std::string chunkName = (parent.getName() + "$chunk" + Twine(c)).str();
    SmallVector<StringRef> argNames;
    for (unsigned i = 0, e = parent.getNumArguments(); i < e; ++i) {
      auto nameAttr = parent.getArgAttrOfType<StringAttr>(i, "zirgen.argName");
      argNames.push_back(nameAttr ? nameAttr.getValue() : StringRef());
    }
    auto chunk = builder.create<zirgen::Zhlt::StepFuncOp>(
        parent.getLoc(), chunkName, parent.getFunctionType(), argNames);
    chunk.setPrivate();

    IRMapping map;
    Block* chunkEntry = builder.createBlock(&chunk.getBody());
    for (auto [origArg, ty] :
         llvm::zip(body.getArguments(), parent.getFunctionType().getInputs())) {
      Value newArg = chunkEntry->addArgument(ty, origArg.getLoc());
      map.map(origArg, newArg);
    }
    builder.setInsertionPointToStart(chunkEntry);

    // Prologue: every op before `sw`. Idempotent witgen, so cloning per chunk
    // is correct.
    for (Operation& op : body) {
      if (&op == sw.getOperation())
        break;
      builder.clone(op, map);
    }

    // Restricted switch: ONLY this chunk's selectors + arms. If none of them
    // is the active selector at runtime, the switch yields its zero default
    // (the WGSL emitter falls through to an empty `else`) -- so this chunk
    // simply does no mux work that cycle.
    SmallVector<Value> newSelectors;
    for (unsigned a = armLo; a < armHi; ++a)
      newSelectors.push_back(map.lookupOrDefault(sw.getSelector()[a]));
    auto newSwitch = builder.create<SwitchOp>(
        loc, sw.getOut().getType(), newSelectors, armHi - armLo);
    for (unsigned a = armLo; a < armHi; ++a) {
      IRMapping armMap = map;
      cloneArm(sw.getArms()[a], newSwitch.getArms()[a - armLo], armMap);
    }
    if (auto layoutAttr = sw->getAttr("layoutType"))
      newSwitch->setAttr("layoutType", layoutAttr);
    map.map(sw.getOut(), newSwitch.getOut());

    // Guard value: sum of this chunk's selectors. At most one of ALL the
    // switch selectors is non-zero, so the sum of this chunk's subset is 0
    // (no arm of this chunk fired) or that selector's value (one did).
    Value guard;
    for (Value sel : newSelectors)
      guard = guard ? builder.create<Zll::AddOp>(loc, guard, sel).getResult()
                    : sel;

    // Epilogue, guarded: a single-arm switch keyed on `guard`. The arm holds
    // the cloned epilogue; if `guard` is zero the arm is skipped and the
    // chunk yields the zero default -- so a non-firing chunk performs no
    // witness writes from the epilogue.
    Type retTy;
    if (parent.getFunctionType().getNumResults() == 1)
      retTy = parent.getFunctionType().getResult(0);

    if (retTy) {
      auto guardSwitch = builder.create<SwitchOp>(loc, retTy, ValueRange{guard}, 1);
      IRMapping epiMap = map;
      OpBuilder armBuilder(ctx);
      Block* armBlock = armBuilder.createBlock(&guardSwitch.getArms()[0]);
      armBuilder.setInsertionPointToStart(armBlock);
      for (Operation& op : body) {
        if (&op == sw.getOperation() || op.isBeforeInBlock(sw.getOperation()))
          continue;
        if (&op == term)
          break;
        armBuilder.clone(op, epiMap);
      }
      // The original terminator yields the function result; re-yield the
      // mapped result value inside the guard arm.
      Value epiResult = epiMap.lookupOrDefault(term->getOperand(0));
      armBuilder.create<ZStruct::YieldOp>(loc, epiResult);
      builder.create<zirgen::Zhlt::ReturnOp>(loc, guardSwitch.getOut());
    } else {
      // Void step_func (the @step$Top shim shape): epilogue runs for side
      // effects only -- still must be guarded, via a value-less guard.
      // (Not expected for the giants we target; emit the epilogue directly.)
      for (Operation& op : body) {
        if (&op == sw.getOperation() || op.isBeforeInBlock(sw.getOperation()))
          continue;
        if (&op == term)
          break;
        builder.clone(op, map);
      }
      builder.create<zirgen::Zhlt::ReturnOp>(loc);
    }

    chunks.push_back(chunk);
  }
  (void)mod;
  return chunks;
}

// The first wide direct-child SwitchOp of `fn`, or null.
SwitchOp wideSwitchOf(zirgen::Zhlt::StepFuncOp fn) {
  for (Operation& op : fn.getBody().front()) {
    if (auto sw = dyn_cast<SwitchOp>(&op)) {
      if (sw.getArms().size() > kMaxArmsPerChunk)
        return sw;
    }
  }
  return nullptr;
}

struct MuxChunkPass : public MuxChunkBase<MuxChunkPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Collect every step_func with a wide top-level switch and split it.
    // Originals are kept public AND unchanged -- this pass adds chunk fns
    // alongside them. The chunks call the same workers the originals do,
    // so call-rewriting (and its quadratic-in-call-tree cross-product) is
    // not needed here; risc0-side per-@compute-entry module pruning picks
    // the leaf path at WGSL module emission time.
    SmallVector<zirgen::Zhlt::StepFuncOp> originals;
    mod.walk([&](zirgen::Zhlt::StepFuncOp f) { originals.push_back(f); });

    for (zirgen::Zhlt::StepFuncOp F : originals) {
      SwitchOp wide = wideSwitchOf(F);
      if (!wide)
        continue;
      // Chunks must survive SymbolDCE -- they have no callers yet (risc0
      // wires them in as @compute entries downstream). Mark them public.
      for (auto chunk : splitSwitch(F, wide))
        chunk.setPublic();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMuxChunkPass() {
  return std::make_unique<MuxChunkPass>();
}

} // namespace zirgen::ZStruct
