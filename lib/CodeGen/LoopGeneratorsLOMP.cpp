//===------ LoopGeneratorsLOMP.cpp -  IR helper to create loops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGeneratorsLOMP.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

/// Scheduling types of parallel OMP for loops.
/// (Subset taken from OpenMP's enum in kmp.h: sched_type)
enum SchedulingType {
  kmp_sch_static_chunked = 33,
  kmp_sch_static = 34, /**< static unspecialized */
  kmp_sch_dynamic_chunked = 35,
  kmp_sch_guided_chunked = 36 /**< guided unspecialized */
};

static cl::opt<SchedulingType> PollyScheduling(
    "polly-lomp-scheduling",
    cl::desc("Scheduling type of parallel OMP for loops"),
    cl::values(clEnumVal(kmp_sch_static_chunked, "Static chunked"),
               clEnumVal(kmp_sch_static, "Static unspecialized (default)"),
               clEnumVal(kmp_sch_dynamic_chunked, "Dynamic chunked"),
               clEnumVal(kmp_sch_guided_chunked,
                         "Guided chunked (Static + Dynamic)")),
    cl::Hidden, cl::init(kmp_sch_static), cl::Optional, cl::cat(PollyCategory));

static cl::opt<int>
    PollyChunkSize("polly-lomp-chunksize",
                   cl::desc("Chunksize to use by the KMPC runtime calls"),
                   cl::Hidden, cl::init(1), cl::Optional,
                   cl::cat(PollyCategory));

// We generate a loop of either of the following structures:
//
//              BeforeBB                      BeforeBB
//                 |                             |
//                 v                             v
//              GuardBB                      PreHeaderBB
//              /      |                         |   _____
//     __  PreHeaderBB  |                        v  \/    |
//    /  \    /         |                     HeaderBB  latch
// latch  HeaderBB      |                        |\       |
//    \  /    \         /                        | \------/
//     <       \       /                         |
//              \     /                          v
//              ExitBB                         ExitBB
//
// depending on whether or not we know that it is executed at least once. If
// not, GuardBB checks if the loop is executed at least once. If this is the
// case we branch to PreHeaderBB and subsequently to the HeaderBB, which
// contains the loop iv 'polly.indvar', the incremented loop iv
// 'polly.indvar_next' as well as the condition to check if we execute another
// iteration of the loop. After the loop has finished, we branch to ExitBB.
// We expect the type of UB, LB, UB+Stride to be large enough for values that
// UB may take throughout the execution of the loop, including the computation
// of indvar + Stride before the final abort.

void ParallelLoopGeneratorLOMP::createCallSpawnThreads(Value *SubFn,
                                                       Value *SubFnParam,
                                                       Value *LB, Value *UB,
                                                       Value *Stride) {
  const std::string Name = "__kmpc_fork_call";
  Function *F = M->getFunction(Name);
  Type *Kmpc_MicroTy = M->getTypeByName("kmpc_micro");

  if (!Kmpc_MicroTy) {
    // void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
    Type *MicroParams[] = {Builder.getInt32Ty()->getPointerTo(),
                           Builder.getInt32Ty()->getPointerTo()};

    Kmpc_MicroTy = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
  }

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Kmpc_MicroTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *task = Builder.CreatePointerBitCastOrAddrSpaceCast(
      SubFn, Kmpc_MicroTy->getPointerTo());

  Value *Args[] = {
      SourceLocationInfo, Builder.getInt32(4), task, LB, UB, Stride,
      SubFnParam};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::deployParallelExecution(Value *SubFn,
                                                        Value *SubFnParam,
                                                        Value *LB, Value *UB,
                                                        Value *Stride) {

  // Inform OpenMP runtime about the number of threads if non-zero
  if (!(NumberOfThreads->isZero())) {
    Value *gtid = createCallGlobalThreadNum();
    createCallPushNumThreads(gtid, NumberOfThreads);
  }

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
}

std::vector<Type *> ParallelLoopGeneratorLOMP::createSubFnParamList() {
  std::vector<Type *> Arguments(2, Builder.getInt32Ty()->getPointerTo());
  Arguments.insert(Arguments.end(), 3, LongType);
  Arguments.push_back(Builder.getInt8PtrTy());

  return Arguments;
}

void ParallelLoopGeneratorLOMP::createSubFnParamNames(
    Function::arg_iterator AI) {
  AI->setName("polly.kmpc.global_tid");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.bound_tid");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.lb");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.ub");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.inc");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.shared");
}

Value *ParallelLoopGeneratorLOMP::createSubFn(Value *StrideNotUsed,
                                              AllocaInst *StructData,
                                              SetVector<Value *> Data,
                                              ValueMapT &Map,
                                              Function **SubFnPtr) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *IDPtr, *ID, *IV, *pIsLast, *pStride;
  Value *LB, *UB, *Stride, *Shared, *Chunk, *hasWork, *hasIteration;
  Value *adjUB;

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();
  int align = (is64bitArch) ? 8 : 4;
  int chunksize = (PollyChunkSize > 0) ? PollyChunkSize : 1;

  // Store the previous basic block.
  PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  CheckNextBB = BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  PreHeaderBB = BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  pIsLast = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr,
                                 "polly.par.lastIterPtr");
  pStride = Builder.CreateAlloca(LongType, nullptr, "polly.par.StridePtr");

  // Get iterator for retrieving the parameters
  Function::arg_iterator AI = SubFn->arg_begin();
  // First argument holds global thread id. Then move iterator to LB.
  IDPtr = &*AI;
  std::advance(AI, 2);
  LB = &*AI;
  std::advance(AI, 1);
  UB = &*AI;
  std::advance(AI, 1);
  Stride = &*AI;
  std::advance(AI, 1);
  Shared = &*AI;

  UserContext = Builder.CreateBitCast(Shared, StructData->getType(),
                                      "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);

  ID = Builder.CreateAlignedLoad(IDPtr, align, "polly.par.global_tid");

  Builder.CreateAlignedStore(LB, LBPtr, align);
  Builder.CreateAlignedStore(UB, UBPtr, align);
  Builder.CreateAlignedStore(Builder.getInt32(0), pIsLast, align);
  Builder.CreateAlignedStore(Stride, pStride, align);

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  adjUB = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                            "polly.indvar.UBAdjusted");

  Chunk = ConstantInt::get(LongType, chunksize);

  if (isDynamicSchedule) {
    // "DYNAMIC" scheduling types are handled below
    UB = adjUB;
    createCallDispatchInit(ID, LB, UB, Stride, Chunk);
    hasWork = createCallDispatchNext(ID, pIsLast, LBPtr, UBPtr, pStride);
    hasIteration =
        Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, hasWork,
                           Builder.getInt32(1), "polly.hasIteration");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(CheckNextBB);
    hasWork = createCallDispatchNext(ID, pIsLast, LBPtr, UBPtr, pStride);
    hasIteration =
        Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, hasWork,
                           Builder.getInt32(1), "polly.hasWork");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(PreHeaderBB);
    LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
    UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");
  } else {
    // "STATIC" scheduling types are handled below
    createCallStaticInit(ID, pIsLast, LBPtr, UBPtr, pStride, Chunk);

    LB = Builder.CreateAlignedLoad(LBPtr, align, "polly.indvar.init");
    UB = Builder.CreateAlignedLoad(UBPtr, align, "polly.indvar.UB");

    hasWork = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, UB, adjUB,
                                 "polly.UB_slt_adjUB");

    UB = Builder.CreateSelect(hasWork, UB, adjUB);
    Builder.CreateAlignedStore(UB, UBPtr, align);

    hasIteration = Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE, LB,
                                      UB, "polly.hasIteration");
    Builder.CreateCondBr(hasIteration, PreHeaderBB, ExitBB);

    Builder.SetInsertPoint(CheckNextBB);
    Builder.CreateBr(ExitBB);

    Builder.SetInsertPoint(PreHeaderBB);
  }

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  IV = createLoop(LB, UB, Stride, Builder, LI, DT, AfterBB, ICmpInst::ICMP_SLE,
                  nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  if (!isDynamicSchedule) {
    createCallStaticFini(ID);
  }
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  return IV;
}

Value *ParallelLoopGeneratorLOMP::createCallGlobalThreadNum() {
  const std::string Name = "__kmpc_global_thread_num";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo};
  Value *retVal = Builder.CreateCall(F, Args);

  return retVal;
}

void ParallelLoopGeneratorLOMP::createCallPushNumThreads(Value *global_tid,
                                                         Value *num_threads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, global_tid, num_threads};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallStaticInit(Value *global_tid,
                                                     Value *pIsLast, Value *pLB,
                                                     Value *pUB, Value *pStride,
                                                     Value *Chunk) {

  const std::string Name =
      is64bitArch ? "__kmpc_for_static_init_8" : "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo,
                   global_tid,
                   ScheduleType,
                   pIsLast,
                   pLB,
                   pUB,
                   pStride,
                   ConstantInt::get(LongType, 1),
                   Chunk};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallStaticFini(Value *global_tid) {
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty()};
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, global_tid};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallDispatchInit(Value *global_tid,
                                                       Value *LB, Value *UB,
                                                       Value *Inc,
                                                       Value *Chunk) {

  const std::string Name =
      is64bitArch ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {
      SourceLocationInfo, global_tid, ScheduleType, LB, UB, Inc, Chunk};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLOMP::createCallDispatchNext(Value *global_tid,
                                                         Value *pIsLast,
                                                         Value *pLB, Value *pUB,
                                                         Value *pStride) {

  const std::string Name =
      is64bitArch ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, global_tid, pIsLast, pLB, pUB, pStride};

  Value *retVal = Builder.CreateCall(F, Args);
  return retVal;
}

// FIXME: This function only creates a source location dummy.
GlobalVariable *ParallelLoopGeneratorLOMP::createSourceLocation() {
  const std::string Name = ".loc.dummy";
  GlobalVariable *sourceLocDummy = M->getGlobalVariable(Name);

  if (sourceLocDummy == nullptr) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    // If the ident_t StructType is not available, declare it.
    // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
    if (!identTy) {
      Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt8PtrTy()};

      identTy = StructType::create(M->getContext(), loc_members,
                                   "struct.ident_t", false);
    }

    int strLen = 23;
    auto arrayType = llvm::ArrayType::get(Builder.getInt8Ty(), strLen);

    // Global Variable Definitions
    GlobalVariable *strVar = new GlobalVariable(
        *M, arrayType, true, GlobalValue::PrivateLinkage, 0, ".str.ident");
    strVar->setAlignment(1);

    sourceLocDummy = new GlobalVariable(
        *M, identTy, true, GlobalValue::PrivateLinkage, nullptr, ".loc.dummy");
    sourceLocDummy->setAlignment(8);

    // Constant Definitions
    Constant *initStr = ConstantDataArray::getString(
        M->getContext(), "Source location dummy.", true);

    Value *strPtr = Builder.CreateInBoundsGEP(
        arrayType, strVar, {Builder.getInt32(0), Builder.getInt32(0)});

    Constant *locInitStruct = ConstantStruct::get(
        identTy, {Builder.getInt32(0), Builder.getInt32(0), Builder.getInt32(0),
                  Builder.getInt32(0), (Constant *)strPtr});

    // Initialize variables
    strVar->setInitializer(initStr);
    sourceLocDummy->setInitializer(locInitStruct);
  }

  return sourceLocDummy;
}

void ParallelLoopGeneratorLOMP::collectSchedulingInfo() {
  // Store information so it is available later on
  // 33: kmp_sch_static_chunked, 34: kmp_sch_static
  // 35: kmp_sch_dynamic_chunked, 36: kmp_sch_guided_chunked
  isDynamicSchedule =
      (PollyScheduling < SchedulingType::kmp_sch_dynamic_chunked) ? false
                                                                  : true;
  ScheduleType = Builder.getInt32(PollyScheduling);
}
