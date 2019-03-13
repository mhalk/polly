//===- LoopGeneratorsLOMP.h - IR helper to create loops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LOOP_GENERATORS_LOMP_H
#define POLLY_LOOP_GENERATORS_LOMP_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {
class Value;
class Pass;
class BasicBlock;
} // namespace llvm

namespace polly {
using namespace llvm;

/// This ParallelLoopGenerator subclass handles the generation of parallelized
/// code, utilizing the LLVM OpenMP library.
class ParallelLoopGeneratorKMP : public ParallelLoopGenerator {
public:
  /// Create a parallel loop generator for the current function.
  ParallelLoopGeneratorKMP(PollyIRBuilder &Builder, LoopInfo &LI,
                           DominatorTree &DT, const DataLayout &DL)
      : ParallelLoopGenerator(Builder, LI, DT, DL) {
    is64bitArch = (LongType->getIntegerBitWidth() == 64);
    SourceLocationInfo = createSourceLocation();
    collectSchedulingInfo();
  }

protected:
  /// True if 'LongType' is 64bit wide, otherwise: False.
  bool is64bitArch;

  /// The schedule type, used to execute the microtasks.
  Value *ScheduleType;

  /// True if the ScheduleType is dynamic, false if it is static.
  bool isDynamicSchedule;

  /// The source location struct of this loop.
  /// ident_t = type { i32, i32, i32, i32, i8* }
  GlobalValue *SourceLocationInfo;

  /// Gather information on the scheduling (ScheduleType & isDynamicSchedule).
  void collectSchedulingInfo();

public:
  // The functions below may be used if one does not want to generate a
  // specific OpenMP parallel loop, but generate individual parts of it
  // (e.g. the subfunction definition).

  /// Create a runtime library call to spawn the worker threads.
  ///
  /// @param SubFn      The subfunction which holds the loop body.
  /// @param SubFnParam The parameter for the subfunction (basically the struct
  ///                   filled with the outside values).
  /// @param LB         The lower bound for the loop we parallelize.
  /// @param UB         The upper bound for the loop we parallelize.
  /// @param Stride     The stride of the loop we parallelize.
  void createCallSpawnThreads(Value *SubFn, Value *SubFnParam, Value *LB,
                              Value *UB, Value *Stride);

  void deployParallelExecution(Value *SubFn, Value *SubFnParam, Value *LB,
                               Value *UB, Value *Stride) override;

  std::vector<Type *> createSubFnParamList() const override;

  void createSubFnParamNames(Function::arg_iterator AI) const override;

  Value *createSubFn(Value *Stride, AllocaInst *Struct,
                     SetVector<Value *> UsedValues, ValueMapT &VMap,
                     Function **SubFn) override;

  /// Create a runtime library call to get the current global thread number.
  ///
  /// @return A Value ref which holds the current global thread number.
  Value *createCallGlobalThreadNum();

  /// Create a runtime library call to request a number of threads.
  /// Which will be used in the next OpenMP section (by the next fork).
  ///
  /// @param global_tid   The global thread ID.
  /// @param num_threads  The number of threads to use.
  void createCallPushNumThreads(Value *global_tid, Value *num_threads);

  /// Create a runtime library call to prepare the OpenMP runtime.
  /// For dynamically scheduled loops, saving the loop arguments.
  ///
  /// @param global_tid  The global thread ID.
  /// @param LB          The loop's lower bound.
  /// @param UB          The loop's upper bound.
  /// @param Inc         The loop increment.
  /// @param Chunk       The chunk size of the parallel loop.
  void createCallDispatchInit(Value *global_tid, Value *LB, Value *UB,
                              Value *Inc, Value *Chunk);

  /// Create a runtime library call to retrieve the next (dynamically)
  /// allocated chunk of work for this thread.
  ///
  /// @param global_tid  The global thread ID.
  /// @param pIsLast     Pointer to a flag, which is set to 1 if this is
  ///                    the last chunk of work, or 0 otherwise.
  /// @param pLB         Pointer to the lower bound for the next chunk of work.
  /// @param pUB         Pointer to the upper bound for the next chunk of work.
  /// @param pStride     Pointer to the stride for the next chunk of work.
  ///
  /// @return A Value which holds 1 if there is work to be done, 0 otherwise.
  Value *createCallDispatchNext(Value *global_tid, Value *pIsLast, Value *pLB,
                                Value *pUB, Value *pStride);

  /// Create a runtime library call to prepare the OpenMP runtime.
  /// For statically scheduled loops, saving the loop arguments.
  ///
  /// @param global_tid  The global thread ID.
  /// @param pIsLast     Pointer to a flag, which is set to 1 if this is
  ///                    the last chunk of work, or 0 otherwise.
  /// @param pLB         Pointer to the lower bound for the next chunk of work.
  /// @param pUB         Pointer to the upper bound for the next chunk of work.
  /// @param pStride     Pointer to the stride for the next chunk of work.
  /// @param Chunk       The chunk size of the parallel loop.
  void createCallStaticInit(Value *global_tid, Value *pIsLast, Value *pLB,
                            Value *pUB, Value *pStride, Value *Chunk);

  /// Create a runtime library call to mark the end of
  /// a statically scheduled loop.
  ///
  /// @param global_tid  The global thread ID.
  void createCallStaticFini(Value *global_tid);

  /// Create the current source location.
  ///
  /// Known issue: Generates only(!) dummy values.
  GlobalVariable *createSourceLocation();
};
} // end namespace polly
#endif
