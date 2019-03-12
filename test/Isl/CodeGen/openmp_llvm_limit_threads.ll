; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -S < %s | FileCheck %s --check-prefix=AUTO
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -polly-num-threads=1 -S < %s | FileCheck %s --check-prefix=ONE
; RUN: opt %loadPolly -polly-codegen -polly-parallel -polly-omp-backend=LLVM -polly-num-threads=4 -S < %s | FileCheck %s --check-prefix=FOUR
;
; This is a copy of the test case 'openmp_limit_threads',
; adapted for the LLVM OpenMP backend.

; In automatic mode, no threads are pushed explicitly.
; AUTO-NOT: call void @__kmpc_push_num_threads
; ONE: call void @__kmpc_push_num_threads(%struct.ident_t* @.loc.dummy{{[.0-9]*}}, i32 %{{[0-9]+}}, i32 1)
; FOUR: call void @__kmpc_push_num_threads(%struct.ident_t* @.loc.dummy{{[.0-9]*}}, i32 %{{[0-9]+}}, i32 4)
;
;    void storePosition(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          A[i + j * 1024] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @storePosition(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc4 ], [ 0, %entry ]
  %exitcond5 = icmp ne i64 %indvars.iv3, 1024
  br i1 %exitcond5, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = shl nsw i64 %indvars.iv, 10
  %tmp6 = add nsw i64 %indvars.iv3, %tmp
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp6
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret void
}
