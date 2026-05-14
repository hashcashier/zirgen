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

#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/stats/OpStats.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;
namespace cl = llvm::cl;

namespace zirgen {
namespace codegen {

namespace {

void addCommonSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<StringAttr>(
      [](CodegenEmitter& cg, auto strAttr) { cg.emitEscapedString(strAttr); });
  opts.addLiteralSyntax<IntegerAttr>(
      [](CodegenEmitter& cg, auto intAttr) { cg << intAttr.getValue().getZExtValue(); });
}

void addCppSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << "Val(" << elems[0] << ")";
    } else if (elems.size() == 4) {
      cg << "ExtVal(";
      cg.interleaveComma(elems);
      cg << ")";
    }
  });
}

void addRustSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<PolynomialAttr>([&](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << "Val::new(" << elems[0] << ")";
    } else {
      cg << "ExtVal::new(";
      cg.interleaveComma(elems, [&](auto elem) { cg << "Val::new(" << elem << ")"; });
      cg << ")";
    }
  });
}

// BabyBear Montgomery encoding, computed at codegen time. risc0's field
// elements are stored in Montgomery form (R = 2^32), and the WGSL prelude's
// `mul` is a Montgomery multiplication -- so a field literal, which the IR
// carries in direct form, must be emitted pre-encoded. Mirrors `encode` / `mul`
// in risc0_core's baby_bear.rs exactly: encode(a) = mul(R2, a % P).
static uint32_t babyBearMontgomeryEncode(uint64_t directValue) {
  constexpr uint32_t kP = 2013265921u;  // 15 * 2^27 + 1
  constexpr uint32_t kM = 0x88000001u;  // -P^-1 mod 2^32
  constexpr uint32_t kR2 = 1172168163u; // R^2 mod P, R = 2^32
  uint32_t a = static_cast<uint32_t>(directValue % kP);
  uint64_t o64 = static_cast<uint64_t>(kR2) * static_cast<uint64_t>(a);
  uint32_t low = 0u - static_cast<uint32_t>(o64);
  uint32_t red = kM * low;
  o64 += static_cast<uint64_t>(red) * static_cast<uint64_t>(kP);
  uint32_t ret = static_cast<uint32_t>(o64 >> 32);
  return ret >= kP ? ret - kP : ret;
}

// WGSL has no operator overloading, so the BabyBear field operators that C++
// renders as infix `+ - *` (via the CodegenInfixOp trait) must instead become
// calls to prelude helper functions: add/sub/mul for base Val, ext_add/ext_sub/
// ext_mul for the degree-4 extension field. Registering an op-syntax handler
// here intercepts the op before its default infix emitExpr runs (see
// CodegenEmitter::emitExpr). Field literals are emitted as Montgomery-form u32.
void addWgslSyntax(CodegenOptions& opts) {
  opts.addLiteralSyntax<PolynomialAttr>([](CodegenEmitter& cg, auto polyAttr) {
    auto elems = polyAttr.asArrayRef();
    if (elems.size() == 1) {
      cg << babyBearMontgomeryEncode(elems[0]) << "u";
    } else {
      // ExtVal is `alias ExtVal = vec4<u32>` in the prelude; each component is
      // an independently Montgomery-encoded base-field element.
      cg << "ExtVal(";
      cg.interleaveComma(elems,
                         [&](auto elem) { cg << babyBearMontgomeryEncode(elem) << "u"; });
      cg << ")";
    }
  });

  // A RefAttr is a column reference. C++ relies on Reg's implicit int ctor
  // (`/*offset=*/N`); WGSL has no implicit conversions, so emit an explicit
  // `Reg(Nu)` construction (Reg is `struct Reg { col: u32 }` in the prelude).
  opts.addLiteralSyntax<ZStruct::RefAttr>([](CodegenEmitter& cg, ZStruct::RefAttr refAttr) {
    cg << "Reg(" << refAttr.getIndex() << "u)";
  });

  auto isExtVal = [](mlir::Type ty) {
    auto vt = llvm::dyn_cast<Zll::ValType>(ty);
    return vt && bool(vt.getExtended());
  };
  auto binOp = [isExtVal](llvm::StringRef valFn, llvm::StringRef extFn) {
    return [=](CodegenEmitter& cg, auto op) {
      llvm::StringRef fn = isExtVal(op->getResult(0).getType()) ? extFn : valFn;
      cg << EmitPart(fn) << "(" << op->getOperand(0) << ", " << op->getOperand(1) << ")";
    };
  };
  opts.addOpSyntax<Zll::AddOp>(binOp("add", "ext_add"));
  opts.addOpSyntax<Zll::SubOp>(binOp("sub", "ext_sub"));
  opts.addOpSyntax<Zll::MulOp>(binOp("mul", "ext_mul"));

  // --- Layout / buffer ops -------------------------------------------------
  // WGSL has no generics, so the CUDA `BoundLayout<T>` template is
  // monomorphized: WgslLanguageSyntax::emitLayoutDef emits a companion
  // `struct BoundLayout_<T> { layout: T, buf: u32 }` for every layout type T.
  // These op handlers thread the runtime `.buf` field through layout
  // navigation and construct the right monomorphized wrapper. emitInvokeMacro
  // cannot do this (it never sees the result type), but addOpSyntax handlers
  // get the typed op. The base/ref expression is emitted more than once; for
  // codegen'd SSA values it is a saved variable name, so this is correct if
  // occasionally verbose. TODO(wgsl): bind to a `let` to avoid re-emission.

  // bind_layout(LAYOUT_CONST, buffer) -> BoundLayout_<T>(LAYOUT_CONST, buffer)
  opts.addOpSyntax<ZStruct::BindLayoutOp>([](CodegenEmitter& cg, ZStruct::BindLayoutOp op) {
    auto symAttr = llvm::cast<mlir::FlatSymbolRefAttr>(op.getLayoutAttr());
    cg << "BoundLayout_" << cg.getTypeName(op.getType()) << "("
       << CodegenIdent<IdentKind::Const>(symAttr.getAttr()) << ", " << op.getBuffer() << ")";
  });

  // layoutLookup: narrow the layout to a member field, keep the buffer.
  opts.addOpSyntax<ZStruct::LookupOp>([](CodegenEmitter& cg, ZStruct::LookupOp op) {
    CodegenIdent<IdentKind::Field> member(op.getMemberAttr());
    if (llvm::isa<ZStruct::LayoutType, ZStruct::LayoutArrayType>(op.getBase().getType())) {
      cg << "BoundLayout_" << cg.getTypeName(op.getOut().getType()) << "(" << op.getBase()
         << ".lyt." << member << ", " << op.getBase() << ".buf)";
    } else {
      cg << op.getBase() << "." << member;
    }
  });

  // layoutSubscript: index a layout array, keep the buffer. A Val index is in
  // Montgomery form and must be decoded to a plain u32 first.
  opts.addOpSyntax<ZStruct::SubscriptOp>([](CodegenEmitter& cg, ZStruct::SubscriptOp op) {
    auto emitIndex = [&cg, &op]() {
      if (llvm::isa<Zll::ValType>(op.getIndex().getType()))
        cg << "decode(" << op.getIndex() << ")";
      else
        cg << op.getIndex();
    };
    if (llvm::isa<ZStruct::LayoutArrayType>(op.getBase().getType())) {
      cg << "BoundLayout_" << cg.getTypeName(op.getOut().getType()) << "(" << op.getBase()
         << ".lyt[" << EmitPart(emitIndex) << "], " << op.getBase() << ".buf)";
    } else {
      cg << op.getBase() << "[" << EmitPart(emitIndex) << "]";
    }
  });

  // load(reg, distance): reg is a BoundLayout_Reg; reg.layout.col is the column
  // and reg.buf the buffer id. load/load_ext/load_as_ext mirror LoadOp::emitExpr.
  opts.addOpSyntax<ZStruct::LoadOp>([](CodegenEmitter& cg, ZStruct::LoadOp op) {
    bool resultExt = bool(op.getType().getExtended());
    bool refExt = bool(op.getRef().getType().getElement().getExtended());
    if (refExt)
      cg << "load_ext(";
    else if (resultExt)
      cg << "load_as_ext(";
    else
      cg << "load(";
    cg << op.getRef() << ".lyt.col, " << op.getRef() << ".buf, " << op.getDistance() << ")";
  });

  // store(reg, val): reg is a BoundLayout_Reg.
  opts.addOpSyntax<ZStruct::StoreOp>([](CodegenEmitter& cg, ZStruct::StoreOp op) {
    if (op.getVal().getType().getFieldK() > 1)
      cg << "store_ext(";
    else
      cg << "store(";
    cg << op.getRef() << ".lyt.col, " << op.getRef() << ".buf, " << op.getVal() << ")";
  });

  // get_buffer(name): buffers are a small named set; emit a u32 buffer id that
  // the WGSL prelude maps to a @group/@binding storage buffer.
  opts.addOpSyntax<ZStruct::GetBufferOp>([](CodegenEmitter& cg, ZStruct::GetBufferOp op) {
    cg << "buf_" << CodegenIdent<IdentKind::Var>(op.getNameAttr());
  });

  // eqz(val, "loc"): a witness-consistency assertion. WGSL has no strings, so
  // the diagnostic location is dropped; the base/ext split is resolved here
  // (the op carries the type) rather than in emitInvokeMacro, which does not.
  opts.addOpSyntax<Zll::EqualZeroOp>([isExtVal](CodegenEmitter& cg, Zll::EqualZeroOp op) {
    if (isExtVal(op.getIn().getType()))
      cg << "eqz_ext(" << op.getIn() << ")";
    else
      cg << "eqz(" << op.getIn() << ")";
  });

  // invoke_extern(name, args...): the circuit's escape hatch. assert/log/print
  // are pure no-ops in witgen (matching CUDA witgen.h) AND carry WGSL-illegal
  // string-literal operands, so they collapse to a single extern_noop() call.
  // The remaining externs read from the preflight trace; they emit as
  // extern_<name>(operands) -- the `ctx` context arg is dropped, and the
  // prelude supplies an `extern_<name>` helper. Array-returning externs
  // (divide/getMemoryTxn/bigIntExtern/...) return a WGSL array<Val,N>, which
  // emitSaveResults projects per result.
  opts.addOpSyntax<Zll::ExternOp>([](CodegenEmitter& cg, Zll::ExternOp op) {
    // op.getName() is Operation::getName() ("zll.extern"); the extern's own name
    // is the $name string attribute, reached via getNameAttr().
    llvm::StringRef name = op.getNameAttr().getValue();
    if (name.equals_insensitive("assert") || name.equals_insensitive("log") ||
        name.equals_insensitive("print")) {
      cg << "extern_noop()";
      return;
    }
    cg << "extern_" << CodegenIdent<IdentKind::Func>(op.getNameAttr()) << "(";
    cg.interleaveComma(op.getOperands());
    cg << ")";
  });
}

} // namespace

CodegenOptions getRustCodegenOpts() {
  static codegen::RustLanguageSyntax kRust;
  codegen::CodegenOptions opts(&kRust);
  addCommonSyntax(opts);
  addRustSyntax(opts);
  ZStruct::addRustSyntax(opts);
  Zhlt::addRustSyntax(opts);
  return opts;
}

CodegenOptions getCppCodegenOpts() {
  static codegen::CppLanguageSyntax kCpp;
  codegen::CodegenOptions opts(&kCpp);
  addCommonSyntax(opts);
  addCppSyntax(opts);
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}

CodegenOptions getCudaCodegenOpts() {
  static codegen::CudaLanguageSyntax kCuda;
  codegen::CodegenOptions opts(&kCuda);
  addCommonSyntax(opts);
  addCppSyntax(opts);
  ZStruct::addCppSyntax(opts);
  Zhlt::addCppSyntax(opts);
  return opts;
}

CodegenOptions getWgslCodegenOpts() {
  static codegen::WgslLanguageSyntax kWgsl;
  codegen::CodegenOptions opts(&kWgsl);
  addCommonSyntax(opts);
  addWgslSyntax(opts);
  // Deliberately NOT calling ZStruct::addCppSyntax / Zhlt::addCppSyntax:
  //  * ZStruct::addCppSyntax only registers the C++ RefAttr literal syntax,
  //    which relies on Reg's implicit int constructor -- addWgslSyntax
  //    registers an explicit `Reg(Nu)` form instead.
  //  * Zhlt::addCppSyntax only registers the "ExecContext& ctx" context
  //    argument, which WGSL elides (buffers are module-scope @group/@binding).
  return opts;
}

} // namespace codegen

namespace {

void optimizeSimple(ModuleOp module) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

void optimizeSplit(ModuleOp module, unsigned stage, const StageOptions& opts) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(Zll::createSplitStagePass(stage));
  if (opts.addExtraPasses) {
    opts.addExtraPasses(opm);
  }
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

void optimizePoly(ModuleOp module, const EmitCodeOptions& opts) {
  PassManager pm(module.getContext());
  OpPassManager& opm = pm.nest<func::FuncOp>();
  opm.addPass(Zll::createMakePolynomialPass());
  opm.addPass(createCanonicalizerPass());
  opm.addPass(createCSEPass());
  opm.addPass(Zll::createComputeTapsPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }
}

llvm::StringRef getOutputDir() {
  if (!codegenCLOptions.isConstructed()) {
    throw(std::runtime_error("codegen command line options must be registered"));
  }

  return codegenCLOptions->outputDir;
}

} // namespace

llvm::ManagedStatic<CodegenCLOptions> codegenCLOptions;

class FileEmitter {
public:
  FileEmitter(StringRef path) : path(path) {}

  void emitIR(const std::string& fn, Operation* op) {
    auto ofs = openOutputFile(fn + ".ir");
    op->print(*ofs.get());
  }

  void emitRustStep(const std::string& stage, func::FuncOp func) {
    auto ofs = openOutputFile("rust_step_" + stage + ".cpp");
    createRustStreamEmitter(*ofs)->emitStepFunc(stage, func);
  }

  void emitGpuStep(const std::string& stage, const std::string& suffix, func::FuncOp func) {
    auto ofs = openOutputFile("step_" + stage + suffix);
    createGpuStreamEmitter(*ofs, suffix)->emitStepFunc(stage, func);
  }

  void emitPolyFunc(const std::string& fn, func::FuncOp func) {
    if (codegenCLOptions->validitySplitCount > 1) {
      for (size_t i : llvm::seq(size_t(codegenCLOptions->validitySplitCount))) {
        auto ofs = openOutputFile("rust_" + fn + "_" + std::to_string(i) + ".cpp");
        createRustStreamEmitter(*ofs)->emitPolyFunc(
            fn, func, i, size_t(codegenCLOptions->validitySplitCount));
      }
    } else {
      auto ofs = openOutputFile("rust_" + fn + ".cpp");
      createRustStreamEmitter(*ofs)->emitPolyFunc(fn, func, /*split part=*/0, /*num splits=*/1);
    }
  }

  void emitPolyEdslFunc(func::FuncOp func) {
    auto ofs = openOutputFile("poly_edsl.cpp");
    createCppStreamEmitter(*ofs)->emitPoly(func);
  }

  void emitPolyExtFunc(func::FuncOp func) {
    auto ofs = openOutputFile("poly_ext.rs");
    createRustStreamEmitter(*ofs)->emitPolyExtFunc(func);
  }

  void emitTapsCpp(func::FuncOp func) {
    auto ofs = openOutputFile("taps.cpp");
    createCppStreamEmitter(*ofs)->emitTaps(func);
  }

  void emitTaps(func::FuncOp func) {
    auto ofs = openOutputFile("taps.rs");
    createRustStreamEmitter(*ofs)->emitTaps(func);
  }

  void emitInfo(func::FuncOp func) {
    auto ofs = openOutputFile("info.rs");
    createRustStreamEmitter(*ofs)->emitInfo(func);
  }

  void emitHeader(func::FuncOp func) {
    auto ofs = openOutputFile("impl.h");
    createCppStreamEmitter(*ofs)->emitHeader(func);
  }

  void
  emitEvalCheck(const std::string& suffix, const std::string& headerSuffix, func::FuncOp func) {
    if (codegenCLOptions->validitySplitCount > 1) {
      for (size_t i : llvm::seq(size_t(codegenCLOptions->validitySplitCount))) {
        auto ofs = openOutputFile("eval_check_" + std::to_string(i) + suffix);
        createGpuStreamEmitter(*ofs, suffix)
            ->emitPoly(func, i, size_t(codegenCLOptions->validitySplitCount));
      }
    } else {
      auto ofs = openOutputFile("eval_check" + suffix);
      createGpuStreamEmitter(*ofs, suffix)->emitPoly(func, /*split part=*/0, /*num splits=*/1);
    }

    auto ofs = openOutputFile("eval_check" + headerSuffix);
    createGpuStreamEmitter(*ofs, suffix)->emitPoly(func, 0, 0, /*declsOnly=*/true);
  }

  void emitAllLayouts(mlir::ModuleOp op) {
    emitLayout(op, codegen::getRustCodegenOpts(), ".rs.inc");
    emitLayout(op, codegen::getCppCodegenOpts(), ".cpp.inc");
    emitLayout(op, codegen::getCudaCodegenOpts(), ".cu.inc");
  }

  void emitLayout(mlir::ModuleOp op, const codegen::CodegenOptions& opts, StringRef suffix) {
    auto ofs = openOutputFile(("layout" + suffix).str());
    codegen::CodegenEmitter emitter(opts, ofs.get(), op->getContext());
    op.walk([&](ZStruct::GlobalConstOp constOp) {
      emitter.emitTypeDefs(constOp);
      constOp.emitGlobal(emitter);
    });
  }

private:
  std::string path;

  std::unique_ptr<llvm::raw_ostream> openOutputFile(const std::string& name) {
    std::string filename = path + "/" + name;
    std::error_code ec;
    auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
    if (ec) {
      throw std::runtime_error("Unable to open file: " + filename);
    }
    return ofs;
  }
};

void registerCodegenCLOptions() {
  *codegenCLOptions;
  registerOpStatsCLOptions();
}

void emitCode(ModuleOp module, const EmitCodeOptions& opts) {
  FileEmitter emitter(getOutputDir());
  optimizeSimple(module);
  emitter.emitAllLayouts(module);

  auto stepsAttr = Zll::lookupModuleAttr<Zll::StepsAttr>(module);

  llvm::StringSet seenStages;

  for (auto [stageIndex, stage] : llvm::enumerate(stepsAttr.getSteps())) {
    auto stageOpts = opts.stages.lookup(stage);
    seenStages.insert(stage.strref());
    auto stageName = stage.str();

    auto moduleCopy = dyn_cast<ModuleOp>(module->clone());
    optimizeSplit(moduleCopy, stageIndex, stageOpts);
    moduleCopy.walk([&](func::FuncOp func) {
      std::string outputFile;
      if (stageOpts.outputFile.empty())
        outputFile = stageName;
      else
        outputFile = stageOpts.outputFile;

      emitter.emitRustStep(outputFile, func);
      if (stageName == "compute_accum" || stageName == "verify_accum") {
        emitter.emitGpuStep(outputFile, ".metal", func);
      }
      emitter.emitGpuStep(outputFile, ".cu", func);
    });
  }

  for (auto k : opts.stages.keys()) {
    if (!seenStages.contains(k)) {
      llvm::errs() << "Options specified for stage " << k << " but no stage " << k << " seen\n";
      exit(1);
    }
  }

  optimizePoly(module, opts);

  BogoCycleAnalysis bogoCycles;
  bogoCycles.printStatsIfRequired(module, llvm::outs());

  module.walk([&](func::FuncOp func) {
    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func);
    emitter.emitEvalCheck(".cu", ".cuh", func);
    emitter.emitEvalCheck(".metal", ".h", func);
    emitter.emitPolyEdslFunc(func);
    emitter.emitHeader(func);
    emitter.emitTapsCpp(func);
  });
}

void emitCodeZirgenPoly(ModuleOp module, StringRef outputDir) {
  FileEmitter emitter(outputDir);

  // Inline everything, since everything else expects there to be a single function left.
  PassManager pm(module.getContext());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to apply stage1 passes");
  }

  // Save as IR so we can generate predicates to verify the validity polynomial.
  emitter.emitIR("validity", module);

  BogoCycleAnalysis bogoCycles;
  bogoCycles.printStatsIfRequired(module, llvm::outs());

  module.walk([&](func::FuncOp func) {
    emitter.emitPolyExtFunc(func);
    emitter.emitTaps(func);
    emitter.emitInfo(func);
    emitter.emitTapsCpp(func);
  });

  // Split up functions for poly_fp
  pm.clear();
  pm.addPass(Zll::createBalancedSplitPass(/*maxOps=*/1000));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module))) {
    throw std::runtime_error("Failed to balanced split");
  }

  module.walk([&](func::FuncOp func) {
    if (SymbolTable::getSymbolVisibility(func) == SymbolTable::Visibility::Private)
      return;

    emitter.emitPolyFunc("poly_fp", func);
    emitter.emitEvalCheck(".cu", ".cuh", func);
  });
}

std::string escapeString(llvm::StringRef str) {
  std::string out = "\"";
  for (size_t i = 0; i < str.size(); i++) {
    unsigned char c = str[i];
    if (' ' <= c and c <= '~' and c != '\\' and c != '"') {
      out.push_back(c);
    } else {
      out.push_back('\\');
      switch (c) {
      case '"':
        out.push_back('"');
        break;
      case '\\':
        out.push_back('\\');
        break;
      case '\t':
        out.push_back('t');
        break;
      case '\r':
        out.push_back('r');
        break;
      case '\n':
        out.push_back('n');
        break;
      default:
        char const* const hexdig = "0123456789ABCDEF";
        out.push_back('x');
        out.push_back(hexdig[c >> 4]);
        out.push_back(hexdig[c & 0xF]);
      }
    }
  }
  out.push_back('"');
  return out;
}

} // namespace zirgen
