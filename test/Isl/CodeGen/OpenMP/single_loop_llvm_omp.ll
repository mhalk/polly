; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -polly-omp-backend=LLVM -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -polly-omp-backend=LLVM -polly-lomp-scheduling=kmp_sch_dynamic_chunked -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR-DYNAMIC
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-codegen -polly-omp-backend=LLVM -polly-lomp-scheduling=kmp_sch_dynamic_chunked -polly-lomp-chunksize=4 -S -verify-dom-info < %s | FileCheck %s -check-prefix=IR-DYNAMIC-FOUR
; RUN: opt %loadPolly -polly-parallel -polly-parallel-force -polly-import-jscop -polly-codegen -polly-omp-backend=LLVM -S < %s | FileCheck %s -check-prefix=IR-STRIDE4

; This extensive test case tests the creation of the full set of OpenMP calls
; as well as the subfunction creation using a trivial loop as example.
;
; It is a copy of the test case 'single_loop',
; adapted for the LLVM OpenMP backend.

; #define N 1024
; float A[N];
;
; void single_parallel_loop(void) {
;   for (long i = 0; i < N; i++)
;     A[i] = 1;
; }

; IR: %struct.ident_t = type { i32, i32, i32, i32, i8* }

; IR-LABEL: single_parallel_loop()
; IR-NEXT: entry
; IR-NEXT:   %polly.par.userContext = alloca

; IR-LABEL: polly.parallel.for:
; IR-NEXT:   %polly.par.userContext1 = bitcast {}* %polly.par.userContext to i8*
; IR-NEXT:   call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @.loc.dummy, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i64, i64, i8*)* @single_parallel_loop_polly_subfn to void (i32*, i32*, ...)*), i64 0, i64 1024, i64 1, i8* %polly.par.userContext1)
; IR-NEXT:   br label %polly.exiting

; IR: define internal void @single_parallel_loop_polly_subfn(i32* %polly.kmpc.global_tid, i32* %polly.kmpc.bound_tid, i64 %polly.kmpc.lb, i64 %polly.kmpc.ub, i64 %polly.kmpc.inc, i8* %polly.kmpc.shared)
; IR-LABEL: polly.par.setup:
; IR-NEXT:   %polly.par.LBPtr = alloca i64
; IR-NEXT:   %polly.par.UBPtr = alloca i64
; IR-NEXT:   %polly.par.lastIterPtr = alloca i32
; IR-NEXT:   %polly.par.StridePtr = alloca i64
; IR-NEXT:   %polly.par.userContext = bitcast i8* %polly.kmpc.shared
; IR-NEXT:   %polly.par.global_tid = load i32, i32* %polly.kmpc.global_tid
; IR-NEXT:   store i64 %polly.kmpc.lb, i64* %polly.par.LBPtr
; IR-NEXT:   store i64 %polly.kmpc.ub, i64* %polly.par.UBPtr
; IR-NEXT:   store i32 0, i32* %polly.par.lastIterPtr
; IR-NEXT:   store i64 %polly.kmpc.inc, i64* %polly.par.StridePtr
; IR-NEXT:   %polly.indvar.UBAdjusted = add i64 %polly.kmpc.ub, -1
; IR-NEXT:   call void @__kmpc_for_static_init_{{[4|8]}}(%struct.ident_t* @.loc.dummy{{[.0-9]*}}, i32 %polly.par.global_tid, i32 34, i32* %polly.par.lastIterPtr, i64* %polly.par.LBPtr, i64* %polly.par.UBPtr, i64* %polly.par.StridePtr, i64 1, i64 1)
; IR-NEXT:   %polly.indvar.init = load i64, i64* %polly.par.LBPtr
; IR-NEXT:   %polly.indvar.UB = load i64, i64* %polly.par.UBPtr
; IR-NEXT:   %polly.UB_slt_adjUB = icmp slt i64 %polly.indvar.UB, %polly.indvar.UBAdjusted
; IR-NEXT:   %{{[0-9]+}} = select i1 %polly.UB_slt_adjUB, i64 %polly.indvar.UB, i64 %polly.indvar.UBAdjusted
; IR-NEXT:   store i64 %{{[0-9]+}}, i64* %polly.par.UBPtr
; IR-NEXT:   %polly.hasIteration = icmp sle i64 %polly.indvar.init, %{{[0-9]+}}
; IR:   br i1 %polly.hasIteration, label %polly.par.loadIVBounds, label %polly.par.exit

; IR-LABEL: polly.par.exit:
; IR-NEXT:   call void @__kmpc_for_static_fini(%struct.ident_t* @.loc.dummy, i32 %polly.par.global_tid)
; IR-NEXT:   ret void

; IR-LABEL: polly.par.checkNext:
; IR-NEXT:   br label %polly.par.exit

; IR-LABEL: polly.par.loadIVBounds:
; IR-NEXT:   br label %polly.loop_preheader

; IR-LABEL: polly.loop_exit:
; IR-NEXT:   br label %polly.par.checkNext

; IR-LABEL: polly.loop_header:
; IR-NEXT:   %polly.indvar = phi i64 [ %polly.indvar.init, %polly.loop_preheader ], [ %polly.indvar_next, %polly.stmt.S ]
; IR-NEXT:   br label %polly.stmt.S

; IR-LABEL: polly.stmt.S:
; IR-NEXT:   %[[gep:[._a-zA-Z0-9]*]] = getelementptr [1024 x float], [1024 x float]* {{.*}}, i64 0, i64 %polly.indvar
; IR-NEXT:   store float 1.000000e+00, float* %[[gep]]
; IR-NEXT:   %polly.indvar_next = add nsw i64 %polly.indvar, %polly.kmpc.inc
; IR-NEXT:   %polly.loop_cond = icmp sle i64 %polly.indvar_next, %{{[0-9]+}}
; IR-NEXT:   br i1 %polly.loop_cond, label %polly.loop_header, label %polly.loop_exit

; IR-LABEL: polly.loop_preheader:
; IR-NEXT:   br label %polly.loop_header

; IR: attributes #1 = { "polly.skip.fn" }

; IR-DYNAMIC:   call void @__kmpc_dispatch_init_{{[4|8]}}(%struct.ident_t* @.loc.dummy, i32 %polly.par.global_tid, i32 35, i64 %polly.kmpc.lb, i64 %polly.indvar.UBAdjusted, i64 %polly.kmpc.inc, i64 1)
; IR-DYNAMIC-NEXT:   %{{[0-9]+}} = call i32 @__kmpc_dispatch_next_{{[4|8]}}(%struct.ident_t* @.loc.dummy, i32 %polly.par.global_tid, i32* %polly.par.lastIterPtr, i64* %polly.par.LBPtr, i64* %polly.par.UBPtr, i64* %polly.par.StridePtr)
; IR-DYNAMIC-NEXT:   %polly.hasIteration = icmp eq i32 %{{[0-9]+}}, 1
; IR-DYNAMIC-NEXT:   br i1 %polly.hasIteration, label %polly.par.loadIVBounds, label %polly.par.exit

; IR-DYNAMIC-LABEL: polly.par.exit:
; IR-DYNAMIC-NEXT:   ret void

; IR-DYNAMIC-LABEL: polly.par.checkNext:
; IR-DYNAMIC-NEXT:   %{{[0-9]+}} = call i32 @__kmpc_dispatch_next_{{[4|8]}}(%struct.ident_t* @.loc.dummy, i32 %polly.par.global_tid, i32* %polly.par.lastIterPtr, i64* %polly.par.LBPtr, i64* %polly.par.UBPtr, i64* %polly.par.StridePtr)
; IR-DYNAMIC-NEXT:   %polly.hasWork = icmp eq i32 %{{[0-9]+}}, 1
; IR-DYNAMIC-NEXT:   br i1 %polly.hasWork, label %polly.par.loadIVBounds, label %polly.par.exit

; IR-DYNAMIC-LABEL: polly.par.loadIVBounds:
; IR-DYNAMIC-NEXT:   %polly.indvar.init = load i64, i64* %polly.par.LBPtr
; IR-DYNAMIC-NEXT:   %polly.indvar.UB = load i64, i64* %polly.par.UBPtr
; IR-DYNAMIC-NEXT:   br label %polly.loop_preheader

; IR-DYNAMIC-FOUR:   call void @__kmpc_dispatch_init_{{[4|8]}}(%struct.ident_t* @.loc.dummy, i32 %polly.par.global_tid, i32 35, i64 %polly.kmpc.lb, i64 %polly.indvar.UBAdjusted, i64 %polly.kmpc.inc, i64 4)

; IR-STRIDE4:     call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @.loc.dummy, i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64, i64, i64, i8*)* @single_parallel_loop_polly_subfn to void (i32*, i32*, ...)*), i64 0, i64 1024, i64 4, i8* %polly.par.userContext1)

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 16

define void @single_parallel_loop() nounwind {
entry:
  br label %for.i

for.i:
  %indvar = phi i64 [ %indvar.next, %for.inc], [ 0, %entry ]
  %scevgep = getelementptr [1024 x float], [1024 x float]* @A, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %S, label %exit

S:
  store float 1.0, float* %scevgep
  br label %for.inc

for.inc:
  %indvar.next = add i64 %indvar, 1
  br label %for.i

exit:
  ret void
}
