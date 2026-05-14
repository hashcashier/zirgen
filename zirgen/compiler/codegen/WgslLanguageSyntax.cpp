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

// WGSL (WebGPU Shading Language) backend for the zirgen circuit compiler.
//
// This backend emits witness-generation step functions as WGSL so the risc0
// browser prover can run witgen GPU-resident instead of on the wasm CPU path.
//
// WGSL is a restricted shading language, not a C-family language:
//   * no generics      -> BoundLayout<T> must become a flat u32 offset
//   * no references    -> ExecContext& must be elided; buffers become
//                         module-scope @group/@binding storage buffers
//   * no recursion     -> the circuit call graph must be a DAG (it is)
//   * no Result/throw  -> eqz/assert checks must be dropped or routed to a
//                         debug error-flag buffer
//   * no closures      -> map/reduce must be unrolled or lowered
//   * no tuples        -> multi-result functions must be restructured
//
// THIS ITERATION (SP7 iter 3) emits *syntactically* WGSL output to prove the
// codegen toolchain end-to-end: the C++ compiles, gen_zirgen runs the fourth
// emitTarget(WgslCodegenTarget...) pass without crashing, and a steps.wgsl
// file is produced. The semantic lowerings listed above are NOT done here --
// every place that needs one is marked TODO(wgsl) and emits a best-effort
// placeholder so the module walk completes. The semantic work lands in iter 4.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen::codegen {

namespace {

// Emit the WGSL type reference for an argument, result, struct field, or saved
// value of MLIR type `ty`. WGSL has no generics, so a layout-trait type becomes
// its monomorphized BoundLayout_<T> wrapper (paired with a runtime buffer id);
// a buffer type becomes a plain u32 buffer id; everything else uses the
// declared type name.
void emitTypeRef(CodegenEmitter& cg, mlir::Type ty) {
  if (ty.hasTrait<CodegenLayoutTypeTrait>())
    cg << "BoundLayout_" << cg.getTypeName(ty);
  else if (llvm::isa<BufferType>(ty))
    cg << "u32";
  else
    cg << cg.getTypeName(ty);
}

} // namespace

std::string WgslLanguageSyntax::canonIdent(llvm::StringRef ident, IdentKind kind) {
  // TODO(wgsl): WGSL has a large reserved-word set (incl. `super`, produced from
  // the IR field `_super`) and forbids leading `__` / a lone `_`. iter 4 adds
  // collision-safe escaping. For now, mirror the C++ canonicalization so the
  // skeleton produces stable, mostly-valid identifiers.
  switch (kind) {
  case IdentKind::Var:
  case IdentKind::Field:
  case IdentKind::Func: {
    std::string str = convertToCamelFromSnakeCase(ident);
    if (!str.empty())
      str[0] = llvm::toLower(str[0]);
    return str;
  }
  case IdentKind::Type:
    return convertToCamelFromSnakeCase(ident, /*capitalizeFirst=*/true);
  case IdentKind::Const:
    return "k" + convertToCamelFromSnakeCase(ident, /*capitalizeFirst=*/true);
  case IdentKind::Macro:
    // WGSL has no preprocessor; the "macros" become ordinary snake_case helper
    // functions in the step-template prelude (load, store, load_ext, ...), so
    // they are lowercased rather than upper-cased like the C++ #define names.
    return convertToSnakeFromCamelCase(ident);
  }
  throw(std::runtime_error("Unknown ident kind"));
}

void WgslLanguageSyntax::emitConditional(CodegenEmitter& cg,
                                         CodegenValue condition,
                                         EmitPart emitThen) {
  // WGSL has no implicit int->bool; compare the field value against zero.
  cg << "if ((" << condition << ") != 0u) {\n";
  cg << emitThen;
  cg << "}\n";
}

void WgslLanguageSyntax::emitSwitchStatement(CodegenEmitter& cg,
                                             CodegenIdent<IdentKind::Var> resultName,
                                             mlir::Type resultType,
                                             llvm::ArrayRef<CodegenValue> conditions,
                                             llvm::ArrayRef<EmitArmPartFunc> emitArms) {
  // `var` (mutable) since the arms assign into it; WGSL zero-inits declared vars.
  cg << "var " << resultName << ": ";
  emitTypeRef(cg, resultType);
  cg << ";\n";
  for (const auto& [cond, emitArm] : llvm::zip(conditions, emitArms)) {
    cg << "if ((" << cond << ") != 0u) {\n";
    auto result = emitArm();
    cg << resultName << " = " << result << ";\n";
    cg << "} else ";
  }
  cg << "{\n";
  cg << "  // TODO(wgsl): unreachable mux arm (no assert in WGSL)\n";
  cg << "}\n";
}

void WgslLanguageSyntax::emitFuncDeclaration(CodegenEmitter& cg,
                                             CodegenIdent<IdentKind::Func> funcName,
                                             llvm::ArrayRef<std::string> contextArgDecls,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                             mlir::FunctionType funcType) {
  // WGSL has no forward declarations; functions must be defined before use, in
  // dependency order. (WgslCodegenTarget reports declExt == implExt, so
  // emitTarget never calls this for the step files -- present for completeness.)
}

void WgslLanguageSyntax::emitFuncDefinition(CodegenEmitter& cg,
                                            CodegenIdent<IdentKind::Func> funcName,
                                            llvm::ArrayRef<std::string> contextArgDecls,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                            mlir::FunctionType funcType,
                                            mlir::Region* body) {
  cg << "fn " << funcName << "(";
  if (!contextArgDecls.empty()) {
    // TODO(wgsl): context args arrive as C++ decl strings (e.g. "ExecContext&
    // ctx"). WGSL has no references; iter 4 elides these and rewrites buffer
    // access to module-scope @group/@binding storage buffers. Emit verbatim for
    // now so the function signature is at least present.
    cg.interleaveComma(contextArgDecls, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!argNames.empty())
      cg << ", ";
  }
  cg.interleaveComma(llvm::zip(argNames, funcType.getInputs()), [&](auto nt) {
    CodegenIdent<IdentKind::Var> name = std::get<0>(nt);
    Type ty = std::get<1>(nt);
    cg << name << ": ";
    emitTypeRef(cg, ty);
  });
  cg << ")";

  auto results = funcType.getResults();
  if (results.size() == 1) {
    cg << " -> ";
    emitTypeRef(cg, results[0]);
  } else if (results.size() > 1) {
    // TODO(wgsl): WGSL has no tuples; multi-result functions need restructuring
    // (out-params or a wrapper struct). Emit the first result type for now.
    cg << " -> /* TODO(wgsl): " << results.size() << " results */ ";
    emitTypeRef(cg, results[0]);
  }

  cg << " {\n";
  cg.emitRegion(*body);
  cg << "}\n";
}

void WgslLanguageSyntax::emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) {
  if (values.empty()) {
    cg << "return;\n";
    return;
  }
  cg << "return ";
  if (values.size() > 1) {
    // TODO(wgsl): WGSL has no tuples; return the first value for now.
    cg << "/* TODO(wgsl): " << values.size() << " results */ ";
  }
  cg << values[0] << ";\n";
}

void WgslLanguageSyntax::emitSaveResults(CodegenEmitter& cg,
                                         llvm::ArrayRef<CodegenIdent<IdentKind::Var>> names,
                                         llvm::ArrayRef<mlir::Type> types,
                                         EmitPart emitExpression) {
  if (names.empty()) {
    cg << emitExpression << ";\n";
  } else if (names.size() == 1) {
    // `let` is immutable in WGSL, which matches the SSA values produced here.
    cg << "let " << names[0] << ": ";
    emitTypeRef(cg, types[0]);
    cg << " = " << emitExpression << ";\n";
  } else {
    // TODO(wgsl): no tuple destructuring in WGSL; bind the first name for now.
    cg << "let " << names[0] << ": ";
    emitTypeRef(cg, types[0]);
    cg << " = /* TODO(wgsl): " << names.size() << "-tuple destructure */ " << emitExpression
       << ";\n";
  }
}

void WgslLanguageSyntax::emitSaveConst(CodegenEmitter& cg,
                                       CodegenIdent<IdentKind::Const> name,
                                       CodegenValue value) {
  cg << "const " << name << ": " << cg.getTypeName(value.getType()) << " = " << value << ";\n";
}

void WgslLanguageSyntax::emitConstDecl(CodegenEmitter& cg,
                                       CodegenIdent<IdentKind::Const> name,
                                       Type ty) {
  // WGSL has no forward declarations.
}

void WgslLanguageSyntax::emitCall(CodegenEmitter& cg,
                                  CodegenIdent<IdentKind::Func> callee,
                                  llvm::ArrayRef<std::string> contextArgs,
                                  llvm::ArrayRef<CodegenValue> args) {
  cg << callee << "(";
  if (!contextArgs.empty()) {
    // TODO(wgsl): see emitFuncDefinition -- context args are elided in iter 4.
    cg.interleaveComma(contextArgs, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!args.empty())
      cg << ", ";
  }
  cg.interleaveComma(args);
  cg << ")";
}

void WgslLanguageSyntax::emitInvokeMacro(CodegenEmitter& cg,
                                         CodegenIdent<IdentKind::Macro> callee,
                                         llvm::ArrayRef<StringRef> contextArgs,
                                         llvm::ArrayRef<EmitPart> emitArgs) {
  // WGSL has no preprocessor. The codegen "macros" fall into two groups:
  //
  //  1. Structural macros that are language constructs in WGSL, not calls --
  //     these are special-cased below.
  //  2. Everything else (load / store / load_ext / store_ext / invoke_extern /
  //     bind_layout / ...) becomes a call to a snake_case helper function
  //     provided by the step-template prelude. The prelude functions land in
  //     iter 4b; context args ("ctx") are dropped here because WGSL buffers are
  //     module-scope @group/@binding storage, not threaded through a parameter.
  llvm::StringRef name = callee.strref();

  // layoutLookup(base, a.b.c) -> base.a.b.c  (struct field access path)
  if (name == "layoutLookup") {
    assert(emitArgs.size() == 2);
    cg << emitArgs[0] << "." << emitArgs[1];
    return;
  }
  // layoutSubscript(base, idx) -> base[idx]  (array indexing)
  if (name == "layoutSubscript") {
    assert(emitArgs.size() == 2);
    cg << emitArgs[0] << "[" << emitArgs[1] << "]";
    return;
  }
  // eqz(value, "diagnostic message") -> eqz(value)
  // WGSL has no assert and no string type; the message is dropped.
  // TODO(wgsl): route the failure to a debug error-flag buffer instead.
  if (name == "eqz") {
    assert(emitArgs.size() >= 1);
    cg << "eqz(" << emitArgs[0] << ")";
    return;
  }
  // setField(BabyBear) -> nothing. WGSL has no field-type registration; the
  // prelude hardcodes the BabyBear field arithmetic.
  if (name == "setField") {
    cg << "/* setField */";
    return;
  }

  cg << callee;
  if (emitArgs.empty())
    return;
  cg << "(";
  cg.interleaveComma(emitArgs);
  cg << ")";
}

void WgslLanguageSyntax::emitStructDefImpl(CodegenEmitter& cg,
                                           mlir::Type ty,
                                           llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                           llvm::ArrayRef<mlir::Type> types,
                                           bool layout) {
  cg << "struct " << cg.getTypeName(ty) << " {\n";
  assert(names.size() == types.size());
  for (size_t i = 0; i != names.size(); i++) {
    Type subTy = types[i];
    cg << "  " << names[i] << ": ";
    // A layout-trait member of a *non-layout* struct is a bound layout -- it
    // carries a buffer, so it uses the BoundLayout_<T> wrapper. Within a layout
    // struct the members are plain sub-layouts; the buffer is supplied once,
    // when the whole layout is bound.
    if (!layout && subTy.hasTrait<CodegenLayoutTypeTrait>())
      cg << "BoundLayout_" << cg.getTypeName(subTy);
    else
      cg << cg.getTypeName(subTy);
    cg << ",\n";
  }
  cg << "}\n";
}

void WgslLanguageSyntax::emitStructDef(CodegenEmitter& cg,
                                       mlir::Type ty,
                                       llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                       llvm::ArrayRef<mlir::Type> types) {
  emitStructDefImpl(cg, ty, names, types, /*layout=*/false);
}

void WgslLanguageSyntax::emitStructConstruct(CodegenEmitter& cg,
                                             mlir::Type ty,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                             llvm::ArrayRef<CodegenValue> values) {
  // WGSL struct construction is positional: Name(v0, v1, ...). This relies on
  // `values` arriving in field-declaration order (verified in iter 4).
  cg << cg.getTypeName(ty) << "(";
  cg.interleaveComma(values);
  cg << ")";
}

void WgslLanguageSyntax::emitArrayDef(CodegenEmitter& cg,
                                      mlir::Type ty,
                                      mlir::Type elemType,
                                      size_t numElems) {
  cg << "alias " << cg.getTypeName(ty) << " = array<" << cg.getTypeName(elemType) << ", "
     << numElems << ">;\n";
}

void WgslLanguageSyntax::emitArrayConstruct(CodegenEmitter& cg,
                                            mlir::Type ty,
                                            mlir::Type elemType,
                                            llvm::ArrayRef<CodegenValue> values) {
  cg << cg.getTypeName(ty) << "(";
  cg.interleaveComma(values);
  cg << ")";
}

void WgslLanguageSyntax::emitMapConstruct(CodegenEmitter& cg,
                                          CodegenValue array,
                                          std::optional<CodegenValue> layout,
                                          llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                          mlir::Region& body) {
  // TODO(wgsl): WGSL has no closures or higher-order functions. A surviving
  // MapOp must be unrolled (a ZStruct unroll pass exists) or lowered to an
  // explicit loop before WGSL emission. The body region is intentionally not
  // walked in this skeleton iteration.
  cg << "wgsl_todo_map_construct(" << array << ")";
}

void WgslLanguageSyntax::emitReduceConstruct(CodegenEmitter& cg,
                                             CodegenValue array,
                                             CodegenValue init,
                                             std::optional<CodegenValue> layout,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                             mlir::Region& body) {
  // TODO(wgsl): see emitMapConstruct -- ReduceOp needs unrolling or explicit
  // loop lowering before WGSL emission.
  cg << "wgsl_todo_reduce_construct(" << array << ", " << init << ")";
}

void WgslLanguageSyntax::emitLayoutDef(CodegenEmitter& cg,
                                       mlir::Type ty,
                                       llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                       llvm::ArrayRef<mlir::Type> types) {
  // The layout itself is a plain nested struct of sub-layouts / Reg leaves.
  emitStructDefImpl(cg, ty, names, types, /*layout=*/true);
  // WGSL has no generics, so the CUDA `BoundLayout<T>` template is
  // monomorphized: emit a companion wrapper pairing this layout type with a
  // runtime buffer id. The bind_layout / layoutLookup / layoutSubscript / load
  // / store op handlers in addWgslSyntax construct and thread these.
  cg << "struct BoundLayout_" << cg.getTypeName(ty) << " {\n";
  cg << "  layout: " << cg.getTypeName(ty) << ",\n";
  cg << "  buf: u32,\n";
  cg << "}\n";
}

} // namespace zirgen::codegen
