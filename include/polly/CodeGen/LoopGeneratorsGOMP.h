<<<<<<< HEAD
//===- LoopGenerators.h - IR helper to create loops -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
=======
//===- LoopGeneratorsGOMP.h - IR helper to create loops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
>>>>>>> wip_polly_llvm_openmp_backend_patch2
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LOOP_GENERATORS_GOMP_H
#define POLLY_LOOP_GENERATORS_GOMP_H

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
/// code, utilizing the GNU OpenMP library.
class ParallelLoopGeneratorGOMP : public ParallelLoopGenerator {
public:
  /// Create a parallel loop generator for the current function.
  ParallelLoopGeneratorGOMP(PollyIRBuilder &Builder, LoopInfo &LI,
                            DominatorTree &DT, const DataLayout &DL)
      : ParallelLoopGenerator(Builder, LI, DT, DL) {}

<<<<<<< HEAD
  /// The functions below can be used if one does not want to generate a
  /// specific OpenMP parallel loop, but generate individual parts of it
  /// (e.g., the subfunction definition).
=======
  // The functions below may be used if one does not want to generate a
  // specific OpenMP parallel loop, but generate individual parts of it
  // (e.g. the subfunction definition).
>>>>>>> wip_polly_llvm_openmp_backend_patch2

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

<<<<<<< HEAD
  /// Create the runtime library calls for spawn and join of the worker threads.
  /// Additionally, places a call to the specified subfunction.
  ///
  /// @param SubFn      The subfunction which holds the loop body.
  /// @param SubFnParam The parameter for the subfunction (basically the struct
  ///                   filled with the outside values).
  /// @param LB         The lower bound for the loop we parallelize.
  /// @param UB         The upper bound for the loop we parallelize.
  /// @param Stride     The stride of the loop we parallelize.
  void deployParallelExecution(Value *SubFn, Value *SubFnParam, Value *LB,
                               Value *UB, Value *Stride);

  /// Create the parameter definition for the parallel subfunction.
  std::vector<Type *> createSubFnParamList();

  /// Name the parameters of the parallel subfunction.
  ///
  /// @return A vector containing the types of the subfunction's argument(s).
  void createSubFnParamNames(Function::arg_iterator AI);

  /// Create the parallel subfunction.
  ///
  /// @param Stride The induction variable increment.
  /// @param Struct A struct holding all values in @p Values.
  /// @param Values A set of LLVM-IR Values that should be available in
  ///               the new loop body.
  /// @param VMap   A map to allow outside access to the new versions of
  ///               the values in @p Values.
  /// @param SubFn  The newly created subfunction is returned here.
  ///
  /// @return The newly created induction variable.
  Value *createSubFn(Value *Stride, AllocaInst *Struct,
                     SetVector<Value *> UsedValues, ValueMapT &VMap,
                     Function **SubFn);
=======
  void deployParallelExecution(Value *SubFn, Value *SubFnParam, Value *LB,
                               Value *UB, Value *Stride) override;

  virtual Function *prepareSubFnDefinition(Function *F) const override;

  std::tuple<Value *, Function *> createSubFn(Value *Stride, AllocaInst *Struct,
                                              SetVector<Value *> UsedValues,
                                              ValueMapT &VMap) override;
>>>>>>> wip_polly_llvm_openmp_backend_patch2

  /// Create a runtime library call to join the worker threads.
  void createCallJoinThreads();

  /// Create a runtime library call to get the next work item.
  ///
  /// @param LBPtr A pointer value to store the work item begin in.
  /// @param UBPtr A pointer value to store the work item end in.
  ///
  /// @returns A true value if the work item is not empty.
  Value *createCallGetWorkItem(Value *LBPtr, Value *UBPtr);

  /// Create a runtime library call to allow cleanup of the thread.
  ///
  /// @note This function is called right before the thread will exit the
  ///       subfunction and only if the runtime system depends on it.
  void createCallCleanupThread();
};
} // end namespace polly
#endif
