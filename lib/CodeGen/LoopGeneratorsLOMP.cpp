//===------ LoopGenerators.cpp -  IR helper to create loops ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGenerators.h"
#include "polly/CodeGen/LoopGeneratorsLOMP.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

static cl::opt<int>
    PollyNumThreads("polly-num-threads-lomp",
                    cl::desc("Number of threads to use (0 = auto)"), cl::Hidden,
                    cl::init(0));

static cl::opt<int>
    PollyScheduling("polly-lomp-scheduling",
                    cl::desc("Int representation of the KMPC scheduling"),
                    cl::Hidden, cl::init(34));

static cl::opt<int>
    PollyChunkSize("polly-lomp-chunksize",
                    cl::desc("Chunksize to use by the KMPC runtime calls"),
                    cl::Hidden, cl::init(1));


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
Value *ParallelLoopGeneratorLOMP::createParallelLoop(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {
  Function *SubFn;

  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Stride, Struct, UsedValues, Map, &SubFn);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  // Add one as the upper bound provided by OpenMP is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();

  return IV;
}

void ParallelLoopGeneratorLOMP::createCallSpawnThreads(Value *SubFn,
                                                   Value *SubFnParam, Value *LB,
                                                   Value *UB, Value *Stride) {
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

  Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = {SubFn, SubFnParam, NumberOfThreads, LB, UB, Stride};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLOMP::createCallGetWorkItem(Value *LBPtr,
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

void ParallelLoopGeneratorLOMP::createCallJoinThreads() {
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

void ParallelLoopGeneratorLOMP::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);

    /**
    // ToDo: kmpc adaption

    // Place mandatory "fini" call in case of a static scheduling type.
    if (ScheduleType == 0) { createCallStaticFini(Location, ID); }
    **/
  }

  Builder.CreateCall(F, {});
}

void ParallelLoopGeneratorLOMP::deployParallelExecution(Value *SubFn,
                                                   Value *SubFnParam, Value *LB,
                                                   Value *UB, Value *Stride) {
  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();
}

std::vector<Type *> ParallelLoopGeneratorLOMP::createSubFnParamList() {
  std::vector<Type *> Arguments(1, Builder.getInt8PtrTy());

  /**
  // ToDo: kmpc adaption
  std::vector<Type *> Arguments(2, Builder.getInt32Ty()->getPointerTo());
  Arguments.insert(Arguments.end(), 3, LongType);
  Arguments.push_back(Builder.getInt8PtrTy());
  **/

  return Arguments;
}

void ParallelLoopGeneratorLOMP::createSubFnParamNames(Function::arg_iterator AI) {
  AI->setName("polly.par.userContext");

  /**
  // ToDo: kmpc adaption
  AI->setName("polly.kmpc.global_tid"); std::advance(AI, 1);
  AI->setName("polly.kmpc.bound_tid"); std::advance(AI, 1);
  AI->setName("polly.kmpc.lb"); std::advance(AI, 1);
  AI->setName("polly.kmpc.ub"); std::advance(AI, 1);
  AI->setName("polly.kmpc.inc"); std::advance(AI, 1);
  AI->setName("polly.kmpc.shared");
  **/
}

Value *ParallelLoopGeneratorLOMP::createSubFn(Value *Stride, AllocaInst *StructData,
                                          SetVector<Value *> Data,
                                          ValueMapT &Map, Function **SubFnPtr) {
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
  *SubFnPtr = SubFn;

  return IV;
}

Value *ParallelLoopGeneratorLOMP::createCallGlobalThreadNum(Value *loc) {
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

   Value *Args[] = {loc};
   Value *retVal = Builder.CreateCall(F, Args);

   return retVal;
}

void ParallelLoopGeneratorLOMP::createCallPushNumThreads(Value *loc, Value *id,
                              Value *num_threads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {identTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, id, num_threads};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallStaticInit(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_for_static_init_8" :
                                         "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType, LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // Static schedule
  Value *schedule = Builder.getInt32(34);

  Value *Args[] = {loc, global_tid, schedule, pIsLast, pLB, pUB, pStride,
                   ConstantInt::get(LongType, 1), ConstantInt::get(LongType, 1)};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallStaticFini(Value *loc, Value *id) {
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

  Value *Args[] = {loc, id};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorLOMP::createCallDispatchInit(Value *loc,
                                                   Value *global_tid,
                                                   Value *Sched, Value *LB,
                                                   Value *UB, Value *Inc,
                                                   Value *Chunk) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_dispatch_init_8" :
                                         "__kmpc_dispatch_init_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty(), LongType, LongType,
                      LongType, LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, global_tid, Sched, LB, UB, Inc, Chunk};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorLOMP::createCallDispatchNext(Value *loc,
                                                   Value *global_tid,
                                                   Value *pIsLast, Value *pLB,
                                                   Value *pUB, Value *pStride) {

  bool is64bitArch = (LongType->getIntegerBitWidth() == 64);
  const std::string Name = is64bitArch ? "__kmpc_dispatch_next_8" :
                                         "__kmpc_dispatch_next_4";
  Function *F = M->getFunction(Name);
  StructType *identTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {identTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {loc, global_tid, pIsLast, pLB, pUB, pStride};

  Value *retVal = Builder.CreateCall(F, Args);
  return retVal;
}



// FIXME: This function only creates a location dummy.
GlobalVariable *ParallelLoopGeneratorLOMP::createSourceLocation(Module *M) {
  const std::string Name = ".loc.dummy";
  GlobalVariable *dummy_src_loc = M->getGlobalVariable(Name);

  if (dummy_src_loc == nullptr) {
    StructType *identTy = M->getTypeByName("struct.ident_t");

    // If the ident_t StructType is not available, declare it.
    // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
    if (!identTy) {
      Type *loc_members[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt32Ty(), Builder.getInt32Ty(),
                             Builder.getInt8PtrTy() };

      identTy = StructType::create(M->getContext(), loc_members,
                                   "struct.ident_t", false);
    }

    int strLen = 23;
    auto arrayType = llvm::ArrayType::get(Builder.getInt8Ty(), strLen);

    // Global Variable Definitions
    GlobalVariable* theString = new GlobalVariable(*M, arrayType, true,
                            GlobalValue::PrivateLinkage, 0, ".str.ident");
    theString->setAlignment(1);

    dummy_src_loc = new GlobalVariable(*M, identTy, true,
    GlobalValue::PrivateLinkage, nullptr, ".loc.dummy");
    dummy_src_loc->setAlignment(8);

    // Constant Definitions
    Constant *locInit_str = ConstantDataArray::getString(M->getContext(),
    "Source location dummy.", true);

    Value *stringGEP = Builder.CreateInBoundsGEP(arrayType, theString,
      {Builder.getInt32(0), Builder.getInt32(0)});

    Constant *locInit_struct = ConstantStruct::get(identTy,
      { Builder.getInt32(0), Builder.getInt32(0), Builder.getInt32(0),
        Builder.getInt32(0), (Constant*) stringGEP});

    // Initialize variables
    theString->setInitializer(locInit_str);
    dummy_src_loc->setInitializer(locInit_struct);
  }

  return dummy_src_loc;
}
