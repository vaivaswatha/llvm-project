// RUN: mlir-opt %s -mlir-disable-threading -allow-unregistered-dialect -test-print-value-range-propagation 2>&1 | FileCheck %s

func @abs1(%f: f32) -> f32 {
  %g = math.abs %f : f32
  // CHECK: %0 = math.abs %arg0 : f32 : [0.000000e+00 : f32 ; INF]
  return %g : f32
}

func @abs2(%arg0 : i1) -> f32 {
  cond_br %arg0, ^bb0, ^bb1

^bb0:
  %f = arith.constant -1.0 : f32
  // CHECK: %cst = arith.constant -1.000000e+00 : f32 : [-1.000000e+00 : f32 ; -1.000000e+00 : f32]
  br ^bb2 (%f : f32)

^bb1:
  %g = arith.constant 2.0 : f32
  // CHECK: %cst_0 = arith.constant 2.000000e+00 : f32 : [2.000000e+00 : f32 ; 2.000000e+00 : f32]
  br ^bb2 (%g : f32)

^bb2 (%arg1 : f32):
  %h = math.abs %arg1 : f32
  // CHECK: %1 = math.abs %0 : f32 : [0.000000e+00 : f32 ; 2.000000e+00 : f32]
  return %h : f32
}

func @add1() -> f32 {
  %f = arith.constant -1.0 : f32
  %g = arith.constant 2.0 : f32
  %h = arith.addf %f, %g : f32
  // CHECK: %0 = arith.addf %cst, %cst_0 : f32 : [1.000000e+00 : f32 ; 1.000000e+00 : f32]
  return %h : f32
}

func @add2() -> i32 {
  %f = arith.constant -1 : i32
  %g = arith.constant 2 : i32
  %h = arith.addi %f, %g : i32
  // CHECK: %0 = arith.addi %c-1_i32, %c2_i32 : i32 : [1 : i32 ; 1 : i32]
  return %h : i32
}

func @add3(%arg0 : i1, %arg3 : f32) -> f32 {
  %arg3_abs = math.abs %arg3 : f32
  cond_br %arg0, ^bb0, ^bb1

^bb0:
  %f1 = arith.constant -1.0 : f32
  %f2 = arith.constant -2.0 : f32
  br ^bb2 (%f1,  %f2 : f32, f32)

^bb1:
  %g1 = arith.constant 2.0 : f32
  %g2 = arith.constant 3.0 : f32
  br ^bb2 (%g1, %g2 : f32, f32)

^bb2 (%arg1 : f32, %arg2 : f32):
  %h = arith.addf %arg1, %arg2 : f32
  // CHECK: %2 = arith.addf %0, %1 : f32 : [-3.000000e+00 : f32 ; 5.000000e+00 : f32]
  %i = arith.addf %arg1, %arg3_abs : f32
  // CHECK %4 = arith.addf %1, %0 : f32 : [-1.000000e+00 : f32 ; INF]
  %j = arith.addf %h, %i : f32
  // CHECK %5 = arith.addf %3, %4 : f32 : [-4.000000e+00 : f32 ; INF]
  return %j : f32
}
