<<<<<<< HEAD
//===------ LoopGeneratorsGOMP.cpp -  IR helper to create loops -----------===//
=======
//===------ LoopGeneratorsGOMP.cpp - IR helper to create loops ------------===//
>>>>>>> wip_polly_llvm_openmp_backend_patch2
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
<<<<<<< HEAD
// This file contains functions to create scalar and parallel loops as LLVM-IR.
=======
// This file contains functions to create parallel loops as LLVM-IR.
>>>>>>> wip_polly_llvm_openmp_backend_patch2
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGeneratorsGOMP.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

<<<<<<< HEAD
=======
extern int polly::PollyNumThreads;
extern OMPGeneralSchedulingType polly::PollyScheduling;
extern int polly::PollyChunkSize;

>>>>>>> wip_polly_llvm_openmp_backend_patch2
void ParallelLoopGeneratorGOMP::createCallSpawnThreads(Value *SubFn,
                                                       Value *SubFnParam,
                                                       Value *LB, Value *UB,
                                                       Value *Stride) {
  const std::string Name = "GOMP_parallel_loop_runtime_start";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                          Builder.getVoidTy(), Builder.getInt8PtrTy(), false)),
                      Builder.getInt8PtrTy(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

<<<<<<< HEAD
  Value *Args[] = {SubFn, SubFnParam, NumberOfThreads, LB, UB, Stride};
=======
  Value *Args[] = {SubFn, SubFnParam, Builder.getInt32(PollyNumThreads),
                   LB,    UB,         Stride};
>>>>>>> wip_polly_llvm_openmp_backend_patch2

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorGOMP::deployParallelExecution(Value *SubFn,
                                                        Value *SubFnParam,
                                                        Value *LB, Value *UB,
                                                        Value *Stride) {
  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();
}

<<<<<<< HEAD
std::vector<Type *> ParallelLoopGeneratorGOMP::createSubFnParamList() {
  std::vector<Type *> Arguments(1, Builder.getInt8PtrTy());
  return Arguments;
}

void ParallelLoopGeneratorGOMP::createSubFnParamNames(
    Function::arg_iterator AI) {
  AI->setName("polly.par.userContext");
}

Value *ParallelLoopGeneratorGOMP::createSubFn(Value *Stride,
                                              AllocaInst *StructData,
                                              SetVector<Value *> Data,
                                              ValueMapT &Map,
                                              Function **SubFnPtr) {
=======
Function *ParallelLoopGeneratorGOMP::prepareSubFnDefinition(Function *F) const {
  FunctionType *FT =
      FunctionType::get(Builder.getVoidTy(), {Builder.getInt8PtrTy()}, false);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);
  // Name the function's arguments
  SubFn->arg_begin()->setName("polly.par.userContext");
  return SubFn;
}

// Create a subfunction of the following (preliminary) structure:
//
//    PrevBB
//       |
//       v
//    HeaderBB
//       |   _____
//       v  v    |
//   CheckNextBB  PreHeaderBB
//       |\       |
//       | \______/
//       |
//       v
//     ExitBB
//
// HeaderBB will hold allocations and loading of variables.
// CheckNextBB will check for more work.
// If there is more work to do: go to PreHeaderBB, otherwise go to ExitBB.
// PreHeaderBB loads the new boundaries (& will lead to the loop body later on).
// ExitBB marks the end of the parallel execution.
std::tuple<Value *, Function *>
ParallelLoopGeneratorGOMP::createSubFn(Value *Stride, AllocaInst *StructData,
                                       SetVector<Value *> Data,
                                       ValueMapT &Map) {
  if (PollyScheduling != OMPGeneralSchedulingType::runtime) {
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the scheduling type 'runtime'.\n";
  }

>>>>>>> wip_polly_llvm_openmp_backend_patch2
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *Ret1, *HasNextSchedule, *LB, *UB, *IV;
  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

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
  UserContext = Builder.CreateBitCast(
      &*SubFn->arg_begin(), StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Ret1 = createCallGetWorkItem(LBPtr, UBPtr);
  HasNextSchedule = Builder.CreateTrunc(Ret1, Builder.getInt1Ty(),
                                        "polly.par.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, PreHeaderBB, ExitBB);

  // Add code to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(PreHeaderBB);
  LB = Builder.CreateLoad(LBPtr, "polly.par.LB");
  UB = Builder.CreateLoad(UBPtr, "polly.par.UB");

  // Subtract one as the upper bound provided by OpenMP is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  IV = createLoop(LB, UB, Stride, Builder, LI, DT, AfterBB, ICmpInst::ICMP_SLE,
                  nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
<<<<<<< HEAD
  *SubFnPtr = SubFn;

  return IV;
=======

  return std::make_tuple(IV, SubFn);
>>>>>>> wip_polly_llvm_openmp_backend_patch2
}

Value *ParallelLoopGeneratorGOMP::createCallGetWorkItem(Value *LBPtr,
                                                        Value *UBPtr) {
  const std::string Name = "GOMP_loop_runtime_next";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {LongType->getPointerTo(), LongType->getPointerTo()};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(
      Return, Builder.CreateZExt(Builder.getFalse(), Return->getType()));
  return Return;
}

void ParallelLoopGeneratorGOMP::createCallJoinThreads() {
  const std::string Name = "GOMP_parallel_end";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}

void ParallelLoopGeneratorGOMP::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}
